"""KMAP/Level outliner tree."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from src.core.level import KMapProject


class ModuleEditorOutliner(QtWidgets.QTreeWidget):
    itemSelected = QtCore.Signal(str)
    itemsSelected = QtCore.Signal(object)
    actionRequested = QtCore.Signal(str, str)
    itemRenamed = QtCore.Signal(str, str)

    CONTEXT_ACTIONS = (
        ("Add Module", "add_module", "mapStudioOutlinerAddModuleAction"),
        ("Add Room", "add_room", "mapStudioOutlinerAddRoomAction"),
        ("Add Blueprint", "add_blueprint", "mapStudioOutlinerAddBlueprintAction"),
        ("Add Camera", "add_camera", "mapStudioOutlinerAddCameraAction"),
        ("Add Light", "add_light", "mapStudioOutlinerAddLightAction"),
        ("Rename", "rename", "mapStudioOutlinerRenameAction"),
        ("Duplicate", "duplicate", "mapStudioOutlinerDuplicateAction"),
        ("Delete", "delete", "mapStudioOutlinerDeleteAction"),
        ("Focus in Viewport", "focus_in_viewport", "mapStudioOutlinerFocusViewportAction"),
        ("Validate Selected", "validate_selected", "mapStudioOutlinerValidateSelectedAction"),
        ("Reveal Source File", "reveal_source_file", "mapStudioOutlinerRevealSourceAction"),
    )

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ModuleEditorOutliner")
        self.setAccessibleName("Map Studio project outliner")
        self.setAccessibleDescription(
            "Shows Maya-style scene objects plus modules, rooms, walkmeshes, authored placements, lights, blueprints, and resources in the current KMAP project."
        )
        self.setToolTip(
            "Outliner workflow: select scene objects, double-click to rename, then duplicate/delete/focus selected items through the context menu or workflow panel."
        )
        self.setColumnCount(2)
        self.setHeaderLabels(["Scene Object", "Type"])
        # Stretch the name column and pin Type to its contents so neither
        # ellipsizes into "Scen..." / "Auth..." in a narrow dock.
        header = self.header()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(False)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.itemSelectionChanged.connect(self._selection_changed)
        self.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        self.itemChanged.connect(self._item_changed)
        self._project: KMapProject | None = None

    def set_project(
        self,
        project: KMapProject,
        authored_gameplay_placements=(),
        authored_room_lights=(),
        authored_room_primitives=(),
        authored_room_resrefs=(),
    ) -> None:
        self._project = project
        self.blockSignals(True)
        self.clear()
        root = self._item(project.name, project.project_id, "project", type_text="KMAP")
        root.setExpanded(True)
        self.addTopLevelItem(root)
        scene_objects = self._category("Scene Objects")
        scene_objects.setExpanded(True)
        root.addChild(scene_objects)
        cameras = self._category("Cameras")
        scene_objects.addChild(cameras)
        for camera_name in ("persp", "top", "front", "side"):
            cameras.addChild(
                self._item(
                    camera_name,
                    f"viewport_camera:{camera_name}",
                    "viewport_camera",
                    type_text="Camera",
                    editable=False,
                )
            )
        authored_rooms = self._category("Authored Rooms")
        authored_rooms.setExpanded(True)
        scene_objects.addChild(authored_rooms)
        primitive_rows_by_room: dict[str, list[object]] = {}
        for row in authored_room_primitives or ():
            room = str(getattr(row, "room_resref", "") or "").strip()
            if not room:
                continue
            primitive_rows_by_room.setdefault(room, []).append(row)
        room_resrefs = {
            str(value or "").strip()
            for value in tuple(authored_room_resrefs or ())
            if str(value or "").strip()
        }
        room_resrefs.update(primitive_rows_by_room)
        source_labels = {
            room.get("room_resref"): room.get("metadata", {}).get("source_metadata", {}).get("label")
            for room in (project.extra_sections.get("authored_module") or {}).get("rooms", [])
        }
        for room_resref in sorted(room_resrefs):
            source_label = source_labels.get(room_resref)
            room_item = self._item(
                str(source_label or room_resref),
                f"authored_room:{room_resref}",
                "authored_room",
                type_text="Room",
                editable=False,
            )
            if source_label:
                room_item.setToolTip(0, f"{source_label} ({room_resref})")
            room_item.setExpanded(True)
            authored_rooms.addChild(room_item)
            for row in sorted(
                primitive_rows_by_room.get(room_resref, ()),
                key=lambda value: str(getattr(value, "primitive_name", "") or "").lower(),
            ):
                primitive_name = str(getattr(row, "primitive_name", "") or "").strip()
                primitive_type = str(getattr(row, "primitive_type", "") or "primitive").strip()
                if not primitive_name:
                    continue
                room_item.addChild(
                    self._item(
                        primitive_name,
                        authored_primitive_item_id(room_resref, primitive_name),
                        "authored_primitive",
                        type_text=primitive_type or "Primitive",
                        room_resref=room_resref,
                        primitive_name=primitive_name,
                    )
                )
        modules = self._category("Modules")
        root.addChild(modules)
        for module in project.modules:
            mod_item = self._item(module.module_name, module.module_id, "module", type_text="Module")
            modules.addChild(mod_item)
            rooms = self._category("Rooms")
            mod_item.addChild(rooms)
            for room_id in module.rooms:
                room = project.find_room(room_id)
                if room is not None:
                    rooms.addChild(self._item(room.name, room.room_id, "room", type_text="Room"))
            woks = self._category("Walkmeshes")
            mod_item.addChild(woks)
            for wok_id in module.walkmeshes:
                wok = project.find_walkmesh(wok_id)
                if wok is not None:
                    woks.addChild(self._item(PathText(wok.source_path, wok.wok_id), wok.wok_id, "walkmesh", type_text="WOK"))
        loose_rooms = self._category("Loose Rooms")
        root.addChild(loose_rooms)
        module_room_ids = {room_id for module in project.modules for room_id in module.rooms}
        for room in project.rooms:
            if room.room_id not in module_room_ids:
                loose_rooms.addChild(self._item(room.name, room.room_id, "room", type_text="Room"))
        for label, rows, kind, key in (
            ("Blueprints", project.blueprints, "blueprint", "blueprint_id"),
            ("Lights", project.lights, "light", "id"),
            ("Cameras", project.cameras, "camera", "id"),
            ("Sequences", project.sequences, "sequence", "id"),
            ("Materials", project.materials, "material", "material_id"),
            ("Textures", project.textures, "texture", "texture_id"),
        ):
            cat = self._category(label)
            root.addChild(cat)
            for row in rows:
                item_id = getattr(row, key, "") if not isinstance(row, dict) else str(row.get(key) or row.get("id") or "")
                name = getattr(row, "name", "") if not isinstance(row, dict) else str(row.get("name") or row.get("resref") or item_id)
                cat.addChild(self._item(name or item_id, item_id, kind, type_text=kind.title()))
        authored = self._category("Authored Gameplay")
        root.addChild(authored)
        for placement in authored_gameplay_placements or ():
            placement_id = str(getattr(placement, "placement_id", "") or "")
            kind = str(getattr(placement, "kind", "object") or "object")
            tag = str(getattr(placement, "tag", "") or getattr(placement, "template_resref", "") or placement_id)
            transition = str(getattr(placement, "transition_summary", "") or "")
            is_entry_point = placement_id == "entry_point"
            label = "Player Start" if is_entry_point else f"{kind}: {tag}"
            if transition:
                label = f"{label} ({transition})"
            authored.addChild(
                self._item(
                    label,
                    placement_id,
                    "authored_entry_point" if is_entry_point else "authored_gameplay",
                    type_text="Player Start" if is_entry_point else kind.title(),
                    editable=not is_entry_point,
                )
            )
        authored_lights = self._category("Authored Room Lights")
        root.addChild(authored_lights)
        for light in authored_room_lights or ():
            light_id = str(getattr(light, "light_id", "") or "")
            name = str(getattr(light, "name", "") or light_id)
            room = str(getattr(light, "room_resref", "") or "")
            authored_lights.addChild(self._item(f"{name} ({room})", light_id, "authored_room_light", type_text="Light"))
        root.addChild(self._category("Validation Issues"))
        self.resizeColumnToContents(0)
        self.expandToDepth(1)
        self.blockSignals(False)

    def select_id(self, item_id: str) -> None:
        self.select_ids((item_id,))

    def update_item_text(self, item_id: str, text: str) -> bool:
        """Rename one tree row in place without rebuilding the outliner."""

        wanted = str(item_id or "")
        label = str(text or "").strip()
        if not wanted or not label:
            return False
        blocked = self.blockSignals(True)
        try:
            for item in self.findItems("*", QtCore.Qt.MatchWildcard | QtCore.Qt.MatchRecursive):
                if str(item.data(0, QtCore.Qt.UserRole) or "") != wanted:
                    continue
                item.setText(0, label)
                item.setData(0, QtCore.Qt.UserRole + 4, label)
                kind = str(item.data(0, QtCore.Qt.UserRole + 1) or "")
                item.setToolTip(0, f"{kind}: {label}\nRight-click for Rename, Duplicate, Delete, Focus, and Validate actions.")
                return True
        finally:
            self.blockSignals(blocked)
        return False

    def select_ids(self, item_ids) -> None:
        wanted = {str(value or "") for value in tuple(item_ids or ()) if str(value or "")}
        matches = self.findItems("*", QtCore.Qt.MatchWildcard | QtCore.Qt.MatchRecursive)
        blocked = self.blockSignals(True)
        try:
            self.clearSelection()
            current = None
            for item in matches:
                if str(item.data(0, QtCore.Qt.UserRole) or "") not in wanted:
                    continue
                item.setSelected(True)
                current = item
            if current is not None:
                self.setCurrentItem(current, 0, QtCore.QItemSelectionModel.NoUpdate)
        finally:
            self.blockSignals(blocked)

    def _item(
        self,
        text: str,
        item_id: str,
        kind: str,
        *,
        type_text: str = "",
        room_resref: str = "",
        primitive_name: str = "",
        editable: bool = True,
    ) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([text, type_text])
        item.setData(0, QtCore.Qt.UserRole, item_id)
        item.setData(0, QtCore.Qt.UserRole + 1, kind)
        item.setData(0, QtCore.Qt.UserRole + 2, room_resref)
        item.setData(0, QtCore.Qt.UserRole + 3, primitive_name)
        item.setData(0, QtCore.Qt.UserRole + 4, text)
        item.setToolTip(0, f"{kind}: {text}\nRight-click for Rename, Duplicate, Delete, Focus, and Validate actions.")
        if editable:
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        else:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    def _category(self, text: str) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([text, ""])
        item.setData(0, QtCore.Qt.UserRole + 1, "category")
        item.setToolTip(0, f"{text} category")
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        return item

    def _selection_changed(self) -> None:
        selected_ids = [
            str(item.data(0, QtCore.Qt.UserRole) or "")
            for item in self.selectedItems()
            if str(item.data(0, QtCore.Qt.UserRole) or "")
        ]
        self.itemsSelected.emit(selected_ids)
        if not selected_ids:
            return
        current = self.currentItem()
        item_id = str(current.data(0, QtCore.Qt.UserRole) or "") if current is not None else selected_ids[-1]
        if len(selected_ids) == 1:
            self.itemSelected.emit(item_id or selected_ids[-1])

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802 - Qt API
        """Use Maya-style add/remove clicks instead of Qt's range selection."""

        if event.button() == QtCore.Qt.LeftButton:
            point = event.position().toPoint()
            item = self.itemAt(point)
            modifiers = event.modifiers()
            if item is not None and modifiers & (QtCore.Qt.ShiftModifier | QtCore.Qt.AltModifier):
                blocked = self.blockSignals(True)
                try:
                    if modifiers & QtCore.Qt.AltModifier:
                        item.setSelected(False)
                    else:
                        item.setSelected(True)
                        self.setCurrentItem(item, 0, QtCore.QItemSelectionModel.NoUpdate)
                finally:
                    self.blockSignals(blocked)
                event.accept()
                self._selection_changed()
                return
        super().mousePressEvent(event)

    def _item_changed(self, item: QtWidgets.QTreeWidgetItem, _column: int) -> None:
        item_id = str(item.data(0, QtCore.Qt.UserRole) or "")
        kind = str(item.data(0, QtCore.Qt.UserRole + 1) or "")
        previous = str(item.data(0, QtCore.Qt.UserRole + 4) or "")
        updated = str(item.text(0) or "").strip()
        if item_id and kind != "category":
            if updated and updated != previous:
                item.setData(0, QtCore.Qt.UserRole + 4, updated)
                self.itemRenamed.emit(item_id, updated)

    def _context_menu(self, pos: QtCore.QPoint) -> None:
        item = self.itemAt(pos)
        item_id = str(item.data(0, QtCore.Qt.UserRole) or "") if item else ""
        menu = QtWidgets.QMenu(self)
        menu.setObjectName("mapStudioOutlinerContextMenu")
        for label, action_name, object_name in self.CONTEXT_ACTIONS:
            qaction = menu.addAction(label)
            qaction.setObjectName(object_name)
            qaction.setToolTip(f"{label} for the selected KMAP item")
            qaction.triggered.connect(lambda _checked=False, text=action_name: self.actionRequested.emit(text, item_id))
        menu.exec(self.viewport().mapToGlobal(pos))


def PathText(source_path: str, fallback: str) -> str:
    if not source_path:
        return fallback
    return source_path.replace("\\", "/").rsplit("/", 1)[-1]


def authored_primitive_item_id(room_resref: str, primitive_name: str) -> str:
    return f"authored_primitive:{str(room_resref or '').strip()}:{str(primitive_name or '').strip()}"
