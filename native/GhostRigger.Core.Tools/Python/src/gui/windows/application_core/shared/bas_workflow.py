"""Body Attachment System preview and recipe helpers."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from types import MethodType

try:
    from PySide6 import QtWidgets
except ImportError as exc:  # pragma: no cover - import gate for Qt runtime
    raise RuntimeError("PySide6 is required for the Qt shell") from exc

from src.systems.bas.attachment_alignment import default_bas_attachment_transform
from src.systems.bas.attachment_catalog import repair_bas_body_texture_references
from src.systems.bas.head_resolution import normalize_bas_model_resref, resolve_bas_head_resref
from src.systems.bas.model_recipe import (
    BAS_SLOT_ORDER,
    BAS_SOCKET_BY_SLOT,
    build_bas_model_recipe,
    save_bas_model_recipe,
)
from src.systems.bas.preview_composer import (
    build_bas_preview_model,
    prepare_bas_composed_export_model,
)

log = logging.getLogger(__name__)


class BasWorkflowMixin:
    """Body Attachment System preview and recipe helpers."""

    def _ensure_bas_attachment_catalog(self, *_args) -> None:
        """Lazily build the game-derived BAS item catalog once games resolve."""

        panel = getattr(self, "body_attachment_panel", None)
        if panel is None or not hasattr(panel, "set_attachment_catalog"):
            return
        manager = self._get_resource_manager()
        if manager is None:
            return
        revision = getattr(manager, "revision", None)
        current_catalog = panel.attachment_catalog()
        current_revision_getter = getattr(panel, "attachment_catalog_revision", None)
        current_revision = current_revision_getter() if callable(current_revision_getter) else None
        catalog_ready = current_catalog is not None and (
            revision is None or current_revision == int(revision)
        )
        if not catalog_ready:
            try:
                from src.systems.bas.attachment_catalog import build_bas_attachment_catalog

                catalog = build_bas_attachment_catalog(manager)
            except Exception:
                log.debug("BAS attachment catalog build failed", exc_info=True)
            else:
                if not catalog.empty:
                    panel.set_attachment_catalog(catalog, revision=revision)
                    panel.set_status(
                        "BAS catalog ready: "
                        f"{len(catalog.entries('head'))} K1/K2 head choices and "
                        f"{len(catalog.entries('body'))} headless body choices."
                    )
        if getattr(panel, "selected_mode", lambda: "")() == "character_preset" and hasattr(
            panel, "load_character_preset_catalog"
        ):
            panel.load_character_preset_catalog(manager, revision=revision)

    def _handle_bas_mode_changed(self, mode: str) -> None:
        mode_key = str(mode or "headless_body").strip().lower()
        if mode_key == "character_preset":
            self._ensure_bas_attachment_catalog()
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.set_status(
                    "Character Preset mode: choose an installed UTC to load its canonical body, head, and equipment."
                )
            return
        if mode_key not in {"headless_body", "full_body"}:
            mode_key = "headless_body"
        self._bas_mode = mode_key
        if mode_key == "full_body" and "head" in getattr(self, "_bas_attachments", {}):
            self._bas_attachments.pop("head", None)
            self._bas_attachment_resrefs.pop("head", None)
            self._bas_attachment_transforms.pop("head", None)
            self._current_head_model = None
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.clear_slot_model("head")
        result = self._rebuild_bas_preview()
        if hasattr(self, "body_attachment_panel"):
            self.body_attachment_panel.set_status(result or f"BAS mode: {mode_key.replace('_', ' ')}.")
    def _handle_bas_character_preset_requested(self, preset) -> None:
        """Atomically replace BAS layers with one installed UTC character."""

        panel = getattr(self, "body_attachment_panel", None)
        game = str(getattr(preset, "game", "") or "").strip().upper()
        body_resref = normalize_bas_model_resref(getattr(preset, "body_resref", ""))
        body_game = str(getattr(preset, "body_game", "") or game).strip().upper()
        if panel is None or game not in {"K1", "K2"} or not body_resref:
            if panel is not None:
                panel.set_status("The selected character preset has no loadable body model.")
            return

        body = self._load_bas_attachment_model(body_resref, game=body_game, strict=True)
        if body is None:
            panel.set_status(f"Could not load canonical body {body_game}:{body_resref}.")
            return
        requested_layers: dict[str, str] = dict(getattr(preset, "slot_model_map", {}) or {})
        requested_layer_games: dict[str, str] = dict(
            getattr(preset, "slot_game_map", {}) or {}
        )
        head_resref = normalize_bas_model_resref(getattr(preset, "head_resref", ""))
        if head_resref:
            requested_layers["head"] = head_resref
            requested_layer_games["head"] = str(
                getattr(preset, "head_game", "") or game
            ).strip().upper()
        loaded_layers: dict[str, object] = {}
        missing: list[str] = []
        for slot, resref in requested_layers.items():
            layer_game = str(requested_layer_games.get(slot) or game).upper()
            model = self._load_bas_attachment_model(resref, game=layer_game, strict=True)
            if model is None:
                missing.append(f"{slot}: {layer_game}:{resref}")
                continue
            loaded_layers[slot] = model
        if head_resref and "head" not in loaded_layers:
            head_game = str(requested_layer_games.get("head") or game).upper()
            panel.set_status(
                f"Preset was not applied because canonical head {head_game}:{head_resref} could not be loaded."
            )
            return

        from src.systems.bas.character_presets import apply_bas_texture_override

        apply_bas_texture_override(body, getattr(preset, "body_texture", ""))
        if "head" in loaded_layers:
            apply_bas_texture_override(loaded_layers["head"], getattr(preset, "head_texture", ""))
        setattr(
            body,
            "_gr_bas_character_preset",
            {
                "game": game,
                "utc_resref": str(getattr(preset, "utc_resref", "") or ""),
                "display_name": str(getattr(preset, "display_name", "") or ""),
                "body_game": body_game,
                "head_game": str(getattr(preset, "head_game", "") or game).upper(),
                "slot_games": dict(requested_layer_games),
                "slot_models": dict(getattr(preset, "slot_model_map", {}) or {}),
                "body_texture": str(getattr(preset, "body_texture", "") or ""),
                "head_texture": str(getattr(preset, "head_texture", "") or ""),
                "equipment_items": dict(getattr(preset, "equipment_item_map", {}) or {}),
            },
        )

        self._bas_attachments.clear()
        self._bas_attachment_resrefs.clear()
        self._bas_attachment_transforms.clear()
        self._current_head_model = None
        self._current_attachment_model = None
        self._adopt_bas_body_model(body, body_resref, body_game)
        self._bas_mode = str(getattr(preset, "composition_mode", "headless_body") or "headless_body")
        for slot, model in loaded_layers.items():
            resref = requested_layers[slot]
            self._bas_attachments[slot] = model
            self._bas_attachment_resrefs[slot] = resref
            self._bas_attachment_transforms[slot] = default_bas_attachment_transform(slot, resref)
            if slot == "head":
                self._current_head_model = model
            else:
                self._current_attachment_model = model
        self._bas_active_build_name = f"{getattr(preset, 'utc_resref', body_resref)}_canonical"

        for slot in BAS_SLOT_ORDER:
            if slot != "body":
                panel.clear_slot_model(slot)
        panel.set_composition_mode(self._bas_mode)
        panel.set_mode("character_preset")
        panel.set_body_model(body, resref=body_resref, game=body_game)
        for slot, model in loaded_layers.items():
            panel.set_slot_model(slot, model, resref=requested_layers[slot])
        self._bas_skip_unsupported_preset_layers = True
        try:
            result = self._rebuild_bas_preview()
        finally:
            self._bas_skip_unsupported_preset_layers = False
        skipped_rows = tuple(
            getattr(self, "_bas_last_skipped_attachments", ()) or ()
        )
        skipped_slots = {
            str(row.get("slot", ""))
            for row in skipped_rows
            if isinstance(row, dict)
        }
        for slot in skipped_slots:
            self._bas_attachments.pop(slot, None)
            self._bas_attachment_resrefs.pop(slot, None)
            self._bas_attachment_transforms.pop(slot, None)
            panel.clear_slot_model(slot)
        preset_metadata = getattr(body, "_gr_bas_character_preset", None)
        if isinstance(preset_metadata, dict):
            preset_metadata["nonvisual_slots"] = {
                slot: requested_layers.get(slot, "")
                for slot in sorted(skipped_slots)
                if requested_layers.get(slot, "")
            }
        warning_rows = list(getattr(preset, "warnings", ()) or ())
        if missing:
            warning_rows.append("Missing optional layers: " + ", ".join(missing))
        if skipped_rows:
            warning_rows.append(
                "Non-visual inventory kept as metadata: "
                + ", ".join(
                    f"{row.get('slot', 'item')} ({row.get('socket', 'socket')} unavailable)"
                    for row in skipped_rows
                    if isinstance(row, dict)
                )
            )
        suffix = f" Warnings: {'; '.join(warning_rows[:3])}" if warning_rows else ""
        source_games = {
            source
            for source in (body_game, *requested_layer_games.values())
            if source in {"K1", "K2"} and source != game
        }
        source_note = (
            f" Uses {', '.join(sorted(source_games))} legacy assets."
            if source_games else ""
        )
        panel.set_status(
            f"Loaded canonical {game} character {getattr(preset, 'display_name', '') or getattr(preset, 'utc_resref', '')}. "
            f"{result}{source_note}{suffix}"
        )
        self._refresh_bas_animation_panel_after_layer_change("head")
    def _bas_mode_for_model(self, model) -> str:
        try:
            from src.core.geometry.model_data import CharacterMode, detect_character_mode

            return "headless_body" if detect_character_mode(model) == CharacterMode.HEADLESS_BODY else "full_body"
        except Exception:
            return "headless_body"
    def _handle_bas_attach_requested(self, slot: str, resref: str) -> None:
        slot = str(slot or "").strip().lower()
        resref = str(resref or "").strip()
        if slot in {"left_hand", "right_hand"}:
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.set_status("Hand slots are sockets; attach items through L. Weapon or R. Weapon.")
            return
        if slot == "head" and getattr(self, "_bas_mode", "headless_body") == "full_body":
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.set_status("Full Body BAS mode uses the existing head hooks; attach masks or goggles instead.")
            return
        body = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
        selected_game = self._selected_bas_catalog_game(resref)
        current_game = str(
            getattr(self, "_current_game", "")
            or (self._infer_game_from_model(body) if body is not None else "")
            or "K1"
        ).upper()
        game = selected_game or current_game
        load_resref = normalize_bas_model_resref(resref)
        if slot == "body":
            if not load_resref:
                self.body_attachment_panel.set_status("No headless body model selected.")
                return
            model = self._load_bas_attachment_model(
                load_resref,
                game=game,
                strict=bool(selected_game),
            )
            if model is None:
                self.body_attachment_panel.set_status(f"Could not load {game}:{load_resref}.")
                return
            repair_bas_body_texture_references(
                model,
                manager=self._get_resource_manager(),
                game=game,
                resref=load_resref,
            )
            self._adopt_bas_body_model(model, load_resref, game)
            return
        if body is None:
            self.body_attachment_panel.set_status("No body model loaded.")
            return
        if not resref and slot != "head":
            self.body_attachment_panel.set_status("No attachment model selected.")
            return
        resolution = None
        if slot == "head":
            manager = self._get_resource_manager()
            resolution = resolve_bas_head_resref(
                requested=resref,
                body_model=body,
                manager=manager,
                game=game,
            )
            load_resref = resolution.resolved_resref or load_resref
        if not load_resref:
            self.body_attachment_panel.set_status("No attachment model selected.")
            return
        model = self._load_bas_attachment_model(
            load_resref,
            game=game,
            strict=bool(selected_game),
        )
        if model is None:
            details = ""
            if resolution is not None and resolution.candidates:
                details = f" Tried: {', '.join(resolution.candidates)}."
            self.body_attachment_panel.set_status(f"Could not load {load_resref}.{details}")
            return
        previous_resref = str(self._bas_attachment_resrefs.get(slot, "") or "").strip().lower()
        self._bas_body_model = body
        self._bas_attachments[slot] = model
        self._bas_attachment_resrefs[slot] = load_resref
        if slot not in self._bas_attachment_transforms or previous_resref != load_resref.lower():
            self._bas_attachment_transforms[slot] = default_bas_attachment_transform(slot, load_resref)
        if slot == "head":
            self._current_head_model = model
        else:
            self._current_attachment_model = model
        result = self._rebuild_bas_preview()
        if result:
            self.body_attachment_panel.set_slot_model(slot, model, resref=load_resref)
            suffix = ""
            if resolution is not None and resolution.source and resolution.source != "requested":
                suffix = f" Head resolved from {resolution.source}: {load_resref}."
            elif selected_game:
                suffix = f" Loaded from {selected_game}."
            self.body_attachment_panel.set_status(f"{result}{suffix}")
            self._refresh_bas_animation_panel_after_layer_change(slot)

    def _selected_bas_catalog_game(self, resref: str) -> str:
        panel = getattr(self, "body_attachment_panel", None)
        getter = getattr(panel, "selected_model_game", None)
        selected_getter = getattr(panel, "selected_model_resref", None)
        if not callable(getter) or not callable(selected_getter):
            return ""
        if normalize_bas_model_resref(selected_getter()) != normalize_bas_model_resref(resref):
            return ""
        game = str(getter() or "").strip().upper()
        return game if game in {"K1", "K2"} else ""

    def _adopt_bas_body_model(self, model, resref: str, game: str) -> None:
        """Switch BAS to a loaded body only after strict loading succeeds."""

        self._bas_body_model = model
        self._current_model = model
        self._current_game = str(game or "K1").upper()
        self._model_path = f"{self._current_game}:{resref}"
        self._bas_preview_model = None
        self._bas_active_build_name = ""
        self._bas_mode = self._bas_mode_for_model(model)
        timer = getattr(self, "_animation_timer", None)
        if timer is not None and hasattr(timer, "stop"):
            timer.stop()
        self._animation_engine = None
        self._animation_last_tick = None
        panel = getattr(self, "body_attachment_panel", None)
        if panel is not None:
            panel.set_mode(self._bas_mode)
            panel.set_body_model(model, resref=resref, game=self._current_game)
        if hasattr(self, "_configure_viewport_resources"):
            self._configure_viewport_resources()
        result = self._rebuild_bas_preview()
        if hasattr(self, "properties_panel"):
            self.properties_panel.show_model(model)
        if hasattr(self, "animations_panel"):
            self._load_animation_panel_model(model)
        if hasattr(self, "_populate_animation_library_from_current_model"):
            self._populate_animation_library_from_current_model()
        if hasattr(self, "_update_scene_chrome"):
            self._update_scene_chrome()
        if panel is not None:
            panel.set_status(
                f"Using {self._current_game}:{resref} as the BAS body. {result}"
            )
    def _handle_bas_clear_requested(self, slot: str) -> None:
        slot = str(slot or "").strip().lower()
        if slot in {"body", "left_hand", "right_hand"}:
            return
        self._bas_attachments.pop(slot, None)
        self._bas_attachment_resrefs.pop(slot, None)
        self._bas_attachment_transforms.pop(slot, None)
        self.body_attachment_panel.clear_slot_model(slot)
        if slot == "head":
            self._current_head_model = None
        elif not any(key != "head" for key in self._bas_attachments):
            self._current_attachment_model = None
        result = self._rebuild_bas_preview()
        self.body_attachment_panel.set_status(result or f"Cleared {slot}.")
        self._refresh_bas_animation_panel_after_layer_change(slot)
    def _refresh_bas_animation_panel_after_layer_change(self, slot: str) -> None:
        if not hasattr(self, "animations_panel"):
            return
        selected = ""
        if hasattr(self.animations_panel, "selected_animation"):
            try:
                selected = str(self.animations_panel.selected_animation() or "")
            except Exception:
                selected = ""
        source = self._animation_source_key() if hasattr(self, "_animation_source_key") else "body"
        if source == "body":
            return
        if (source == "head" and slot == "head") or (source == "attachment" and slot != "head"):
            model = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
            self._load_animation_panel_model(model, select_name=selected)
    def _load_bas_attachment_model(self, resref: str, *, game: str = "", strict: bool = False):
        resref = normalize_bas_model_resref(resref)
        if not resref:
            return None
        game = str(
            game
            or getattr(self, "_current_game", "")
            or self._infer_game_from_model(self._bas_body_model or self._current_model)
            or "K1"
        ).upper()
        manager = self._get_resource_manager()
        if manager is None:
            return None
        try:
            if strict and hasattr(manager, "load_model_strict"):
                return manager.load_model_strict(resref, game)
            return manager.load_model(resref, game)
        except Exception:
            log.debug("BAS attachment load failed for %s:%s", game, resref, exc_info=True)
            return None
    def _rebuild_bas_preview(self) -> str:
        body = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
        if body is None:
            return "No body model loaded."
        try:
            previous_preview = getattr(self, "_bas_preview_model", None)
            preview = build_bas_preview_model(
                body_model=body,
                attachment_models=dict(getattr(self, "_bas_attachments", {}) or {}),
                attachment_transforms=dict(getattr(self, "_bas_attachment_transforms", {}) or {}),
                name=str(
                    getattr(self, "_bas_active_build_name", "")
                    or f"{getattr(body, 'name', 'body')}_bas"
                ),
                skip_unsupported=bool(
                    getattr(self, "_bas_skip_unsupported_preset_layers", False)
                ),
            )
            self._bas_last_skipped_attachments = tuple(
                getattr(preview, "_gr_bas_skipped_attachments", ()) or ()
            )
            self._apply_bas_preview_to_viewport(preview, previous_preview=previous_preview)
            skipped_slots = {
                str(row.get("slot", ""))
                for row in self._bas_last_skipped_attachments
                if isinstance(row, dict)
            }
            attached = ", ".join(
                self._bas_attachment_resrefs.get(key, key)
                for key in self._bas_attachments
                if key not in skipped_slots
            )
            return f"BAS preview updated: {attached or 'body only'}."
        except Exception as exc:
            log.exception("BAS preview rebuild failed")
            return f"BAS preview failed: {exc}"
    def _handle_bas_save_build_requested(self) -> None:
        body = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
        if body is None:
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.set_status("No body model loaded.")
            return
        default_name = str(getattr(self, "_bas_active_build_name", "") or f"{getattr(body, 'name', 'body')}_bas")
        name, accepted = QtWidgets.QInputDialog.getText(self, "Save BAS Build", "Model name", text=default_name)
        if not accepted:
            return
        build_name = str(name or "").strip()
        if not build_name:
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.set_status("Save cancelled: model name is required.")
            return
        self._bas_active_build_name = build_name
        rebuild_status = self._rebuild_bas_preview()
        if str(rebuild_status or "").lower().startswith("bas preview failed"):
            if hasattr(self, "body_attachment_panel"):
                self.body_attachment_panel.set_status(rebuild_status)
            return
        path = self._save_bas_model_recipe(body, build_name=build_name)
        if hasattr(self, "body_attachment_panel"):
            if path is not None:
                self.body_attachment_panel.set_status(f"Saved BAS build: {path.name}")
            else:
                self.body_attachment_panel.set_status("Could not save BAS build.")
    def _handle_bas_export_composed_requested(self) -> None:
        """Export the composed BAS preview (body + attached layers) as one model.

        The composed preview is a complete KotorModel — head and other layers
        are real child subtrees under their hook nodes — so it routes through
        the same async MDL/OBJ/FBX workers as any loaded model.  Rigged output
        goes through the ASCII FBX writer (T3502 gate); binary MDL/MDX is the
        game-ready single-model form of the build.
        """

        panel = getattr(self, "body_attachment_panel", None)
        body = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
        if body is None:
            if panel is not None:
                panel.set_status("No body model loaded.")
            return
        rebuild_status = self._rebuild_bas_preview()
        preview = getattr(self, "_bas_preview_model", None)
        if preview is None or str(rebuild_status or "").lower().startswith("bas preview failed"):
            if panel is not None:
                panel.set_status(rebuild_status or "BAS preview is unavailable.")
            return
        export_name = str(
            getattr(self, "_bas_active_build_name", "")
            or f"{getattr(body, 'name', 'body')}_bas"
        )
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Composed BAS Model",
            f"{export_name}.mdl",
            "Binary MDL + MDX (*.mdl);;"
            "OBJ + MTL (*.obj);;"
            "Standard FBX (*.fbx);;"
            "Unity-Compatible FBX (*.fbx);;"
            "Unreal Engine-Compatible FBX (*.fbx);;"
            "3ds Max-Compatible FBX (*.fbx)",
        )
        if not path:
            return
        from src.gui.windows.application_core.shared.model_io import (
            _fbx_compatibility_profile_from_filter,
            _work_export_fbx,
            _work_export_mdl_binary,
            _work_export_obj,
        )

        filter_label = str(selected_filter or "").strip().lower()
        if "fbx" in filter_label:
            export_kind = "fbx"
            path = str(Path(path).with_suffix(".fbx"))
        elif "obj" in filter_label:
            export_kind = "obj"
            path = str(Path(path).with_suffix(".obj"))
        else:
            export_kind = "mdl"
            path = str(Path(path).with_suffix(".mdl"))
        try:
            if export_kind == "mdl":
                # Native MDL preserves the Odyssey DAG exactly, including any
                # legal repeated names. DCC-only global-name normalization must
                # never mutate that game-ready structure.
                export_model = copy.deepcopy(preview)
                self._reset_bas_model_node_traversal(export_model)
                export_report = {"renamed_count": 0, "renamed": []}
            else:
                export_model, export_report = prepare_bas_composed_export_model(
                    preview,
                    require_unique_body_names=True,
                )
            export_model.name = export_name
        except ValueError as exc:
            message = f"Composed export blocked: {exc}"
            if panel is not None:
                panel.set_status(message)
            QtWidgets.QMessageBox.warning(self, "Export Composed BAS Model", message)
            return
        compatibility_profile = _fbx_compatibility_profile_from_filter(selected_filter)
        base_skeleton_model = None
        selected_animation_names = None
        animation_resource_manager = None
        animation_game = ""
        supplemental_animation_models = tuple(
            model
            for model in (getattr(self, "_bas_attachments", {}) or {}).values()
            if model is not None
        )
        if export_kind == "fbx":
            base_resolver = getattr(self, "_fbx_base_skeleton_for_export", None)
            if callable(base_resolver):
                base_skeleton_model = base_resolver(export_model)
            chooser = getattr(self, "_choose_fbx_animation_sets", None)
            if callable(chooser):
                selected_animation_names = chooser(
                    export_model,
                    compatibility_profile,
                    base_skeleton_model=base_skeleton_model,
                    supplemental_models=supplemental_animation_models,
                )
                if selected_animation_names is None:
                    if panel is not None:
                        panel.set_status("Composed FBX export cancelled.")
                    return
            context_getter = getattr(self, "_fbx_resource_context_for_export", None)
            if callable(context_getter):
                animation_resource_manager, animation_game = context_getter(export_model)
        attached = ", ".join(self._bas_attachment_resrefs.get(key, key) for key in self._bas_attachments)

        def _on_complete(result, cancelled=False):
            if cancelled or result is None:
                return
            profile_label = (
                f"{compatibility_profile.replace('_', ' ')} FBX"
                if export_kind == "fbx" else export_kind.upper()
            )
            message = f"Exported composed {profile_label} model ({attached or 'body only'}) -> {Path(path).name}"
            renamed_count = int(export_report.get("renamed_count", 0) or 0)
            if renamed_count:
                message += f"; normalized {renamed_count} colliding attachment node name(s)"
            if panel is not None:
                panel.set_status(message)
            self._log(message, "success")

        if export_kind == "obj":
            self._run_io_async(
                f"Exporting composed OBJ — {Path(path).name}",
                _work_export_obj,
                export_model,
                path,
                tex_cache=self._get_tex_cache_for_export(),
                on_complete=_on_complete,
                error_category="export_error",
            )
        elif export_kind == "fbx":
            self._run_io_async(
                f"Exporting composed FBX — {Path(path).name}",
                _work_export_fbx,
                export_model,
                path,
                tex_cache=self._get_tex_cache_for_export(),
                base_skeleton_model=base_skeleton_model,
                compatibility_profile=compatibility_profile,
                selected_animation_names=selected_animation_names,
                animation_resource_manager=animation_resource_manager,
                animation_game=animation_game,
                supplemental_animation_models=supplemental_animation_models,
                on_complete=_on_complete,
                error_category="export_error",
            )
        else:
            game = (getattr(self, "_current_game", "") or self._infer_game_from_model(body) or "K1").upper()
            self._run_io_async(
                f"Exporting composed binary MDL — {Path(path).name}",
                _work_export_mdl_binary,
                export_model,
                path,
                game_version=game,
                on_complete=_on_complete,
                error_category="export_error",
            )

    def _save_bas_model_recipe(self, body, *, build_name: str = "") -> Path | None:
        try:
            game = (getattr(self, "_current_game", "") or self._infer_game_from_model(body) or "").upper()
            recipe = build_bas_model_recipe(
                body_model=body,
                attachment_models=dict(getattr(self, "_bas_attachments", {}) or {}),
                attachment_resrefs=dict(getattr(self, "_bas_attachment_resrefs", {}) or {}),
                attachment_transforms=dict(getattr(self, "_bas_attachment_transforms", {}) or {}),
                game=game,
                build_name=build_name or getattr(self, "_bas_active_build_name", ""),
                mode=getattr(self, "_bas_mode", "headless_body"),
            )
            path = save_bas_model_recipe(recipe, self.app_root / "src" / "systems" / "bas" / "models")
            self._last_bas_model_recipe_path = path
            log.info("Saved BAS model recipe: %s", path)
            return path
        except Exception:
            log.debug("Could not save BAS model recipe", exc_info=True)
            self._last_bas_model_recipe_path = None
            return None
    def _reset_bas_model_node_traversal(self, model) -> None:
        if model is None:
            return
        base_all_nodes = getattr(type(model), "all_nodes", None)
        if callable(base_all_nodes):
            try:
                setattr(model, "all_nodes", MethodType(base_all_nodes, model))
            except Exception:
                pass
        for attr in ("_gr_original_all_nodes", "_gr_generated_cameras", "_gr_generated_lights"):
            try:
                if hasattr(model, attr):
                    delattr(model, attr)
            except Exception:
                pass
    def _attach_bas_item_to_preview(self, preview, item, socket_name: str, slot: str = "", transform: dict | None = None) -> bool:
        socket = self._find_model_node(preview, socket_name)
        item_copy = copy.deepcopy(item)
        self._reset_bas_model_node_traversal(item_copy)
        item_root = getattr(item_copy, "root_node", None)
        if socket is None or item_root is None:
            return False
        try:
            setattr(item_root, "_gr_bas_attachment_source_model_id", id(item))
            setattr(item_root, "_gr_bas_attachment_source_model_name", str(getattr(item, "name", "") or ""))
            setattr(item_root, "_gr_bas_attachment_source_model_ref", item)
        except Exception:
            pass
        self._prepare_bas_layer_root(item_root, socket, slot or socket_name)
        if not transform:
            item_root.rotation = (0.0, 0.0, 0.0, 1.0)
        if transform:
            self._apply_bas_layer_transform(item_root, transform)
        if slot:
            try:
                self._bas_attachment_transforms[slot] = self._bas_layer_transform_from_model(item_copy)
            except Exception:
                pass
        item_root.parent = socket
        children = getattr(socket, "children", None)
        if children is None:
            socket.children = []
            children = socket.children
        children.append(item_root)
        return True
    def _bas_layer_transform_from_model(self, model) -> dict[str, list[float]]:
        root = getattr(model, "root_node", model)
        position = getattr(root, "position", (0.0, 0.0, 0.0))
        rotation = getattr(root, "rotation", (0.0, 0.0, 0.0, 1.0))
        scale = getattr(root, "scale", (1.0, 1.0, 1.0))
        try:
            scale_values = list(scale)[:3]
        except Exception:
            scale_values = [float(scale or 1.0)] * 3
        return {
            "position": [float(value) for value in list(position)[:3]],
            "rotation": [float(value) for value in list(rotation)[:4]],
            "scale": [float(value) for value in scale_values[:3]],
        }
    def _apply_bas_layer_transform(self, root, transform: dict) -> None:
        if root is None:
            return
        for attr, fallback, count in (
            ("position", (0.0, 0.0, 0.0), 3),
            ("rotation", (0.0, 0.0, 0.0, 1.0), 4),
            ("scale", (1.0, 1.0, 1.0), 3),
        ):
            values = transform.get(attr, fallback) if isinstance(transform, dict) else fallback
            try:
                coerced = [float(value) for value in list(values)[:count]]
            except Exception:
                coerced = []
            while len(coerced) < count:
                coerced.append(float(fallback[len(coerced)]))
            try:
                setattr(root, attr, tuple(coerced))
            except Exception:
                pass
    def _prepare_bas_layer_root(self, item_root, socket, slot: str) -> None:
        socket_name = str(getattr(socket, "name", "") or "").strip()
        if str(slot or "").lower() == "head":
            pos = tuple(float(v) for v in getattr(item_root, "position", (0.0, 0.0, 0.0))[:3])
            try:
                socket_world = socket.world_position()
            except Exception:
                socket_world = (0.0, 0.0, 0.0)
            if abs(float(pos[2]) + float(socket_world[2])) < 0.25 and abs(float(pos[2])) > 0.5:
                item_root.position = (pos[0], pos[1], 0.0)
        setattr(item_root, "_gr_bas_attachment_root", True)
        setattr(item_root, "_gr_bas_attachment_slot", str(slot or "attachment"))
        setattr(item_root, "_gr_bas_socket_name", socket_name)
        self._tag_bas_attachment_subtree(item_root, item_root)
    def _tag_bas_attachment_subtree(self, node, root) -> None:
        stack = [node]
        visited = set()
        while stack:
            current = stack.pop()
            if current is None or id(current) in visited:
                continue
            visited.add(id(current))
            setattr(current, "_gr_bas_attachment_layer", True)
            setattr(current, "_gr_bas_attachment_root_ref", root)
            try:
                setattr(current, "_gr_bas_attachment_source_model_id", int(getattr(root, "_gr_bas_attachment_source_model_id", 0) or 0))
                setattr(current, "_gr_bas_attachment_source_model_name", str(getattr(root, "_gr_bas_attachment_source_model_name", "") or ""))
            except Exception:
                pass
            stack.extend(getattr(current, "children", []) or [])
    def _find_model_node(self, model, name: str):
        target = str(name or "").lower()
        try:
            nodes = model.all_nodes()
        except Exception:
            nodes = []
        by_lower = {str(getattr(node, "name", "") or "").lower(): node for node in nodes}
        for candidate in (target, "rhand_g" if target == "rhand" else "", "lhand_g" if target == "lhand" else ""):
            if candidate and candidate in by_lower:
                return by_lower[candidate]
        return None
    def _bas_socket_for_slot(self, slot: str) -> str:
        slot_key = str(slot or "").strip().lower()
        return BAS_SOCKET_BY_SLOT.get(slot_key, "lhand" if slot_key.startswith("left") else "rhand")
    def _bas_target_scene_object(self, previous_preview=None):
        scene_manager = getattr(self, "scene_manager", None)
        scene = getattr(scene_manager, "active_scene", None)
        if scene is None:
            return None
        body = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
        current_preview = getattr(self, "_bas_preview_model", None)
        candidate_models = [model for model in (body, getattr(self, "_current_model", None), previous_preview, current_preview) if model is not None]
        body_marker = str(id(body)) if body is not None else ""

        for obj in getattr(scene, "objects", []) or []:
            metadata = getattr(obj, "metadata", {}) or {}
            if body_marker and str(metadata.get("_runtime_bas_body_model_id") or "") == body_marker:
                return obj

        selected = scene_manager.get_selected_objects() if hasattr(scene_manager, "get_selected_objects") else []
        for obj in selected:
            runtime = (getattr(obj, "metadata", {}) or {}).get("_runtime_model")
            if any(runtime is model for model in candidate_models):
                return obj

        for obj in getattr(scene, "objects", []) or []:
            runtime = (getattr(obj, "metadata", {}) or {}).get("_runtime_model")
            if any(runtime is model for model in candidate_models):
                return obj
        return None
    def _apply_bas_preview_to_viewport(self, preview, previous_preview=None) -> None:
        target_object = None
        if hasattr(self, "scene_manager"):
            target_object = self._bas_target_scene_object(previous_preview=previous_preview)
        if target_object is not None:
            target_object.metadata["_runtime_model"] = preview
            target_object.metadata["_runtime_bas_body_model"] = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
            target_object.metadata["_runtime_bas_body_model_id"] = str(
                id(getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None))
            )
            target_object.metadata["_runtime_bas_preview_model"] = preview
            target_object.metadata.setdefault("body_attachment_system", {})
            target_object.metadata["body_attachment_system"].update({
                "active": True,
                "mode": getattr(self, "_bas_mode", "headless_body"),
                "attachments": dict(self._bas_attachment_resrefs),
                "layers": [
                    {
                        "slot": slot,
                        "resref": self._bas_attachment_resrefs.get(slot, ""),
                        "enabled": True,
                    }
                    for slot in BAS_SLOT_ORDER
                    if slot in self._bas_attachments
                ],
            })
            self._bas_preview_model = preview
            self.scene_manager.mark_dirty()
            self._refresh_scene_view()
        elif hasattr(self, "viewport"):
            self._bas_preview_model = preview
            self.viewport.load_model(preview)
        else:
            self._bas_preview_model = preview
        self._sync_bas_body_animation_engine()
        self._restore_bas_animation_pose_after_viewport_refresh()
        self._request_bas_viewport_refresh()
        if hasattr(self, "skeleton_panel"):
            self.skeleton_panel.load_model(preview)
        if hasattr(self, "module_geometry_panel"):
            self.module_geometry_panel.show_model(preview)
        if hasattr(self, "sprite_materials_panel"):
            self.sprite_materials_panel.set_model(preview)
    def _sync_bas_body_animation_engine(self, preview=None) -> None:
        try:
            source_key = self._animation_source_key() if hasattr(self, "_animation_source_key") else "body"
        except Exception:
            source_key = "body"
        if source_key != "body":
            return
        body = getattr(self, "_bas_body_model", None) or getattr(self, "_current_model", None)
        if body is None:
            return
        engine = getattr(self, "_animation_engine", None)
        current = getattr(engine, "current_animation", None) if engine is not None else None
        if engine is None or current is None or not hasattr(engine, "model") or getattr(engine, "model", None) is body:
            return
        anim_name = str(getattr(current, "name", "") or "")
        if not anim_name:
            return
        try:
            from src.core.animation.animation_engine import AnimationEngine, SuperModelResolver

            mgr = self._get_resource_manager()
            if mgr is not None:
                SuperModelResolver.configure(mgr)
            t = float(getattr(engine, "current_time", 0.0) or 0.0)
            was_playing = bool(getattr(engine, "is_playing", False))
            loop = bool(getattr(engine, "_loop", getattr(self, "_animation_loop", False)))
            inheritance_game = self._animation_inheritance_game(body)
            inheritance_supermodel = self._animation_inheritance_supermodel(body)
            with self._animation_resolution_context(body, inheritance_game, inheritance_supermodel):
                replacement = AnimationEngine(body)
                if not replacement.play(anim_name, loop=loop, blend=False):
                    return
                replacement.seek(t)
                if not was_playing:
                    replacement.stop()
            self._animation_engine = replacement
        except Exception:
            log.debug("Could not restore BAS animation engine to body model", exc_info=True)
    def _restore_bas_animation_pose_after_viewport_refresh(self) -> None:
        engine = getattr(self, "_animation_engine", None)
        current = getattr(engine, "current_animation", None) if engine is not None else None
        if engine is None or current is None or not hasattr(self, "viewport"):
            return
        try:
            t = float(getattr(engine, "current_time", 0.0) or 0.0)
            pose = engine.evaluate(t)
            if hasattr(self, "_tag_animation_pose_source"):
                pose = self._tag_animation_pose_source(pose, getattr(engine, "model", None))
            self.viewport.set_animation_pose(
                pose,
                name=str(getattr(current, "name", "") or ""),
                time=t,
                length=float(getattr(current, "length", 0.0) or 0.0),
            )
            if hasattr(self.viewport, "set_animation_playback_active"):
                self.viewport.set_animation_playback_active(bool(getattr(engine, "is_playing", False)))
        except Exception:
            log.debug("Could not restore BAS animation pose after viewport refresh", exc_info=True)
    def _request_bas_viewport_refresh(self) -> None:
        viewport = getattr(self, "viewport", None)
        if viewport is None:
            return
        try:
            if hasattr(viewport, "refresh_view"):
                viewport.refresh_view()
            elif hasattr(viewport, "_request_render"):
                viewport._request_render(fast=True, reason="body attachment updated", scene=True, overlay=True, hud=True)
        except Exception:
            log.debug("Could not request BAS viewport refresh", exc_info=True)
