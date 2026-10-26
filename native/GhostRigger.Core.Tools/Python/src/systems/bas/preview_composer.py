"""Runtime preview composition helpers for the Body Attachment System.

BAS preview models keep heads and equipment as socket-following layers. They
must not be merged into the body skin palette or grafted as ordinary body bones.
"""

from __future__ import annotations

import copy
from typing import Any

from src.systems.bas.attachment_alignment import normalize_bas_transform
from src.systems.bas.model_recipe import BAS_SOCKET_BY_SLOT

BAS_ATTACHMENT_SLOTS = (
    "head",
    "mask",
    "goggles",
    "left_weapon",
    "belt",
    "right_weapon",
)
KOTOR_NODE_NAME_LIMIT = 32


def bas_runtime_source_copy_memo(root: Any) -> dict[int, Any]:
    """Keep BAS source models identity-stable while copying scene geometry.

    A BAS layer stores both a runtime source-model reference and its ``id`` so
    renderers can evaluate a matching local attachment pose.  Ordinary
    ``deepcopy`` duplicates that referenced model but preserves the old numeric
    ID, causing the copied scene layer to reject every head-local pose.  Source
    models are resource/runtime owners rather than scene geometry, so scene
    copies must retain those references by identity.
    """

    memo: dict[int, Any] = {}
    stack = [root]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        source_model = getattr(current, "_gr_bas_attachment_source_model_ref", None)
        if source_model is not None:
            memo[id(source_model)] = source_model
        stack.extend(getattr(current, "children", []) or [])
    return memo


def reset_bas_model_node_traversal(model: Any) -> None:
    """Remove stale runtime traversal shims from a copied BAS preview model."""
    if model is None:
        return
    base_all_nodes = getattr(type(model), "all_nodes", None)
    if callable(base_all_nodes):
        try:
            setattr(model, "all_nodes", base_all_nodes.__get__(model, type(model)))
        except Exception:
            pass
    for attr in ("_gr_original_all_nodes", "_gr_generated_cameras", "_gr_generated_lights"):
        try:
            if hasattr(model, attr):
                delattr(model, attr)
        except Exception:
            pass


def find_model_node(model: Any, name: str):
    """Find a node by exact/case-insensitive KOTOR node name."""
    target = str(name or "").strip().lower()
    if not target or model is None:
        return None
    try:
        nodes = model.all_nodes()
    except Exception:
        nodes = []
    by_lower = {str(getattr(node, "name", "") or "").lower(): node for node in nodes}
    for candidate in (
        target,
        "rhand_g" if target == "rhand" else "",
        "lhand_g" if target == "lhand" else "",
    ):
        if candidate and candidate in by_lower:
            return by_lower[candidate]
    return None


def bas_slot_for_preview_socket(socket: str, resref: str = "") -> str:
    """Map Character Builder preview socket choices onto BAS layer slots."""
    key = str(socket or "").strip().lower().replace(" ", "_")
    res = str(resref or "").strip().lower()
    if key in {"head", "headhook"} or res.startswith(("pmh", "pfh", "po_", "n_")) and "head" in key:
        return "head"
    if key in {"lhand", "left_hand", "left_weapon", "lhand_g"} or key.startswith("left"):
        return "left_weapon"
    if key in {
        "rhand",
        "right_hand",
        "right_weapon",
        "rhand_g",
        "lightsaberhook",
        "deflecthook",
    } or key.startswith("right"):
        return "right_weapon"
    return ""


def bas_socket_for_slot(slot: str) -> str:
    """Return the body dummy/socket name used by a BAS slot."""
    return BAS_SOCKET_BY_SLOT.get(str(slot or "").strip().lower(), "")


def tag_bas_attachment_subtree(node: Any, root: Any) -> None:
    """Tag every node in an attachment layer so renderers keep it out of body skinning."""
    stack = [node]
    visited: set[int] = set()
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        setattr(current, "_gr_bas_attachment_layer", True)
        setattr(current, "_gr_bas_attachment_root_ref", root)
        try:
            setattr(
                current,
                "_gr_bas_attachment_source_model_id",
                int(getattr(root, "_gr_bas_attachment_source_model_id", 0) or 0),
            )
            setattr(
                current,
                "_gr_bas_attachment_source_model_name",
                str(getattr(root, "_gr_bas_attachment_source_model_name", "") or ""),
            )
        except Exception:
            pass
        stack.extend(getattr(current, "children", []) or [])


def _apply_bas_layer_transform(root: Any, transform: dict[str, Any] | None) -> None:
    values = normalize_bas_transform(transform)
    for attr in ("position", "rotation", "scale"):
        try:
            setattr(root, attr, tuple(values[attr]))
        except Exception:
            pass


def prepare_bas_layer_root(item_root: Any, socket: Any, slot: str) -> None:
    """Prepare an attachment root for socket-following BAS rendering."""
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
    tag_bas_attachment_subtree(item_root, item_root)


def attach_bas_item_to_preview(
    preview_model: Any,
    item_model: Any,
    socket_name: str,
    *,
    slot: str = "",
    transform: dict[str, Any] | None = None,
) -> bool:
    """Attach a model copy to a body preview as a BAS socket-following layer."""
    socket = find_model_node(preview_model, socket_name)
    if socket is None:
        return False
    item_copy = copy.deepcopy(item_model)
    reset_bas_model_node_traversal(item_copy)
    item_root = getattr(item_copy, "root_node", None)
    if item_root is None:
        return False
    try:
        setattr(item_root, "_gr_bas_attachment_source_model_id", id(item_model))
        setattr(
            item_root,
            "_gr_bas_attachment_source_model_name",
            str(getattr(item_model, "name", "") or ""),
        )
        setattr(item_root, "_gr_bas_attachment_source_model_ref", item_model)
    except Exception:
        pass
    prepare_bas_layer_root(item_root, socket, slot or socket_name)
    _apply_bas_layer_transform(item_root, transform)
    item_root.parent = socket
    children = getattr(socket, "children", None)
    if children is None:
        socket.children = []
        children = socket.children
    children.append(item_root)
    return True


def build_bas_preview_model(
    *,
    body_model: Any,
    attachment_models: dict[str, Any] | None = None,
    attachment_transforms: dict[str, dict[str, Any]] | None = None,
    name: str = "",
    skip_unsupported: bool = False,
) -> Any:
    """Build a copied body preview with BAS attachment layers.

    Stock UTC inventories sometimes contain engine-only equipment for creatures
    whose model has no matching visual socket (for example a turret carrying a
    blaster item but no ``rhand`` node).  Canonical-character workflows may set
    ``skip_unsupported`` so those inventory records remain in preset metadata
    without making the otherwise valid character impossible to preview.  The
    default remains strict for manually authored BAS compositions.
    """
    preview = copy.deepcopy(body_model)
    reset_bas_model_node_traversal(preview)
    attachment_models = attachment_models or {}
    attachment_transforms = attachment_transforms or {}
    skipped: list[dict[str, str]] = []
    for slot in BAS_ATTACHMENT_SLOTS:
        item = attachment_models.get(slot)
        if item is None:
            continue
        socket = bas_socket_for_slot(slot)
        if not attach_bas_item_to_preview(
            preview,
            item,
            socket,
            slot=slot,
            transform=attachment_transforms.get(slot),
        ):
            if skip_unsupported:
                skipped.append(
                    {
                        "slot": slot,
                        "socket": socket,
                        "resref": str(
                            getattr(item, "_gr_source_resref", "")
                            or getattr(item, "name", "")
                            or slot
                        ),
                    }
                )
                continue
            raise ValueError(f"{slot} attachment failed: body has no {socket} socket.")
    setattr(preview, "_gr_bas_skipped_attachments", tuple(skipped))
    if name:
        try:
            preview.name = str(name)
        except Exception:
            pass
    return preview


def prepare_bas_composed_export_model(
    preview_model: Any,
    *,
    require_unique_body_names: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Return an export-safe copy of a composed BAS model.

    KOTOR permits repeated node names in some source assets, while FBX skin,
    hierarchy, and animation connections are globally name-addressed by the
    GhostRigger writer and by common DCC importers.  A composed build can also
    contain the same weapon twice.  Preserve every body name, then rename only
    colliding attachment-layer nodes and rewrite that layer's bone map.
    """

    model = copy.deepcopy(preview_model)
    reset_bas_model_node_traversal(model)
    try:
        nodes = list(model.all_nodes())
    except Exception:
        nodes = []

    def _layer_root(node: Any):
        root = getattr(node, "_gr_bas_attachment_root_ref", None)
        if root is not None:
            return root
        current = node
        visited: set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            if bool(getattr(current, "_gr_bas_attachment_root", False)):
                return current
            current = getattr(current, "parent", None)
        return None

    body_nodes = [node for node in nodes if _layer_root(node) is None]
    attachment_nodes = [node for node in nodes if _layer_root(node) is not None]
    used: set[str] = set()
    rename_by_layer: dict[int, dict[str, str]] = {}
    renamed: list[dict[str, str]] = []
    duplicate_body_names: list[str] = []

    attachment_roots = {
        id(root): root
        for root in (_layer_root(node) for node in attachment_nodes)
        if root is not None
    }
    # Keep detachable head skins animated.  The face shell and its rigid
    # eyes/lids/teeth share the head model's helper-node frame.  Stripping the
    # shell's skin palette freezes the face at the attachment root while those
    # rigid parts continue to follow ``head_g`` and facial controllers; the
    # very first locomotion frame then separates them.  The glTF writer owns a
    # per-mesh Odyssey bind palette, so retaining the source skin is the only
    # representation that preserves both rest registration and animation.
    animated_head_geometry: list[str] = []
    for root in attachment_roots.values():
        slot = str(getattr(root, "_gr_bas_attachment_slot", "") or "").casefold()
        if slot != "head":
            continue
        for child in list(getattr(root, "children", []) or []):
            if not bool(getattr(child, "vertices", None)) or not bool(
                getattr(child, "is_skin", False)
            ):
                continue
            animated_head_geometry.append(str(getattr(child, "name", "") or "head"))
    # Some full-body/NPC MDLs carry a hidden copy of their compatible head's
    # rigid inner geometry (eyes, lids, and teeth). Once BAS supplies the real
    # head, exporting both copies creates the detached second face reported by
    # Unity users. Preserve the body nodes as skeleton/bind nodes, but mark only
    # their replaced hidden geometry for the FBX/OBJ writers to omit.
    head_replacement_names = {
        str(getattr(node, "name", "") or "").casefold()
        for node in attachment_nodes
        if str(getattr(_layer_root(node), "_gr_bas_attachment_slot", "") or "").casefold() == "head"
        and bool(getattr(node, "vertices", None))
    }
    suppressed_body_geometry: list[str] = []
    for node in body_nodes:
        name = str(getattr(node, "name", "") or "")
        if (
            name.casefold() in head_replacement_names
            and getattr(node, "render", True) is False
            and bool(getattr(node, "vertices", None))
        ):
            setattr(node, "_gr_bas_geometry_replaced_by_attachment", True)
            suppressed_body_geometry.append(name)
    scaled_layers: list[str] = []
    for root in attachment_roots.values():
        raw_scale = getattr(root, "scale", (1.0, 1.0, 1.0))
        try:
            scale = tuple(float(value) for value in list(raw_scale)[:3])
        except Exception:
            scale = (1.0, 1.0, 1.0)
        if len(scale) != 3 or any(abs(value - 1.0) > 1e-6 for value in scale):
            slot = str(getattr(root, "_gr_bas_attachment_slot", "") or "attachment")
            scaled_layers.append(f"{slot}={scale}")
    if scaled_layers:
        raise ValueError(
            "FBX/OBJ export cannot safely represent unbaked BAS layer scale; "
            "reset it to 1,1,1 before export (" + ", ".join(scaled_layers) + ")."
        )

    def _unique_name(original: str, slot: str) -> str:
        suffix = f"__{slot or 'attachment'}"
        base_limit = max(1, KOTOR_NODE_NAME_LIMIT - len(suffix))
        candidate = f"{original[:base_limit]}{suffix}"
        serial = 2
        while candidate.lower() in used:
            serial_suffix = f"_{serial}"
            keep = max(1, KOTOR_NODE_NAME_LIMIT - len(suffix) - len(serial_suffix))
            candidate = f"{original[:keep]}{suffix}{serial_suffix}"
            serial += 1
        return candidate

    # The native body DAG is authoritative. Preserve it byte-for-byte for MDL
    # export; DCC callers can request a uniqueness gate instead of silently
    # renaming a body node without identity-based bone/animation provenance.
    for node in body_nodes:
        original = str(getattr(node, "name", "") or "node")
        original_key = original.lower()
        if original_key in used:
            duplicate_body_names.append(original)
        used.add(original_key)

    if require_unique_body_names and duplicate_body_names:
        duplicates = ", ".join(sorted(set(duplicate_body_names), key=str.lower))
        raise ValueError(
            "Unity/DCC export requires unique body node names; "
            f"the native body contains: {duplicates}."
        )

    # An old-name -> new-name bone-map rewrite is safe only when the old name
    # identifies one node inside that attachment layer. Reject ambiguous source
    # layers instead of binding a skin to whichever duplicate was renamed last.
    names_by_layer: dict[int, set[str]] = {}
    duplicate_attachment_names: list[str] = []
    for node in attachment_nodes:
        root = _layer_root(node)
        layer_key = id(root)
        original = str(getattr(node, "name", "") or "node")
        original_key = original.lower()
        layer_names = names_by_layer.setdefault(layer_key, set())
        if original_key in layer_names:
            slot = str(getattr(root, "_gr_bas_attachment_slot", "") or "attachment")
            duplicate_attachment_names.append(f"{slot}:{original}")
        layer_names.add(original_key)
    if duplicate_attachment_names:
        duplicates = ", ".join(sorted(set(duplicate_attachment_names), key=str.lower))
        raise ValueError(
            f"Cannot safely combine an attachment with duplicate node names: {duplicates}."
        )

    for node in attachment_nodes:
        original = str(getattr(node, "name", "") or "node")
        original_key = original.lower()
        if original_key not in used:
            used.add(original_key)
            continue
        root = _layer_root(node)
        slot = str(getattr(root, "_gr_bas_attachment_slot", "") or "attachment").strip().lower()
        replacement = _unique_name(original, slot)
        node.name = replacement
        used.add(replacement.lower())
        if root is not None:
            rename_by_layer.setdefault(id(root), {})[original_key] = replacement
        renamed.append(
            {
                "from": original,
                "to": replacement,
                "slot": slot or "attachment",
                "source_model": str(
                    getattr(root, "_gr_bas_attachment_source_model_name", "") or ""
                ),
            }
        )

    for node in attachment_nodes:
        root = _layer_root(node)
        mapping = rename_by_layer.get(id(root), {}) if root is not None else {}
        bone_map = list(getattr(node, "bone_map", []) or [])
        if bone_map and mapping:
            node.bone_map = [mapping.get(str(name or "").lower(), name) for name in bone_map]

    final_names = [str(getattr(node, "name", "") or "").lower() for node in nodes]
    attachment_layers = []
    for root in attachment_roots.values():
        layer_key = id(root)
        attachment_layers.append(
            {
                "slot": str(getattr(root, "_gr_bas_attachment_slot", "") or "attachment"),
                "source_model": str(
                    getattr(root, "_gr_bas_attachment_source_model_name", "") or ""
                ),
                "renamed_nodes": dict(rename_by_layer.get(layer_key, {})),
            }
        )
    report = {
        "node_count": len(nodes),
        "renamed_count": len(renamed),
        "renamed": renamed,
        "attachment_layers": attachment_layers,
        "rigid_head_geometry": [],
        "animated_head_geometry": animated_head_geometry,
        "suppressed_body_geometry": suppressed_body_geometry,
        "duplicate_body_names": duplicate_body_names,
        "unique_names": len(final_names) == len(set(final_names)),
    }
    try:
        setattr(model, "_gr_bas_export_report", report)
    except Exception:
        pass
    return model, report
