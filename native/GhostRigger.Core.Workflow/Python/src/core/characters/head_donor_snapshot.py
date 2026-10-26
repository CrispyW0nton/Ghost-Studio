"""Immutable native-head donor contracts and structural comparison.

Core Workflow owns the decision that a selected MDL is a usable modular-head
donor.  Core Resources supplies bytes and provenance; this module inspects the
decoded model without reading files, writing binaries, or importing Qt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


HEAD_DONOR_CONTRACT_SCHEMA = "ghostrigger.head_donor_contract"
HEAD_DONOR_CONTRACT_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_WEIGHT_TOLERANCE = 1.0e-4
_FLOAT_TOLERANCE = 1.0e-6
_HEAD_SOCKET_CATEGORIES = {
    "headhook": "body_attachment",
    "neck_g": "head_attachment",
    "maskhook": "headgear",
    "gogglehook": "headgear",
    "revmask1hook": "headgear",
    "revmask2hook": "headgear",
    "camerahook": "camera",
    "freelookhook": "camera",
    "talkdummy": "facial_helper",
}


@dataclass(frozen=True, slots=True)
class HeadDonorNodeContract:
    ordinal: int
    name: str
    parent_ordinal: int | None
    child_ordinals: tuple[int, ...]
    flags: int
    type_label: str
    dense_name_index: int
    sparse_node_number: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    is_mesh: bool
    is_skin: bool
    render: bool
    socket_category: str
    bb_min: tuple[float, float, float]
    bb_max: tuple[float, float, float]
    radius: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeadDonorNodeContract":
        data = dict(payload)
        data["child_ordinals"] = tuple(data.get("child_ordinals") or ())
        for key in ("position", "rotation", "bb_min", "bb_max"):
            data[key] = tuple(data.get(key) or ())
        return cls(**data)


@dataclass(frozen=True, slots=True)
class HeadDonorMeshPayload:
    """Mutable rendered payload facts kept outside the structural fingerprint."""

    node_ordinal: int
    node_name: str
    vertex_count: int
    face_count: int
    normal_count: int
    uv_count: int
    uv2_count: int
    texture: str
    lightmap: str
    geometry_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeadDonorMeshPayload":
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class HeadDonorSkinContract:
    node_ordinal: int
    node_name: str
    parent_ordinal: int | None
    bone_palette: tuple[str, ...]
    bone_node_indices: tuple[int, ...]
    bone_map_floats: tuple[float, ...]
    qbone_rows: tuple[tuple[float, float, float, float], ...]
    tbone_rows: tuple[tuple[float, float, float], ...]
    bind_row_count: int
    skin_row_count: int
    influence_count: int
    max_influences: int
    invalid_influence_rows: int
    unnormalized_rows: int
    influence_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeadDonorSkinContract":
        data = dict(payload)
        data["bone_palette"] = tuple(data.get("bone_palette") or ())
        data["bone_node_indices"] = tuple(
            int(value) for value in data.get("bone_node_indices") or ()
        )
        data["bone_map_floats"] = tuple(
            float(value) for value in data.get("bone_map_floats") or ()
        )
        data["qbone_rows"] = tuple(
            tuple(float(value) for value in row)
            for row in data.get("qbone_rows") or ()
        )
        data["tbone_rows"] = tuple(
            tuple(float(value) for value in row)
            for row in data.get("tbone_rows") or ()
        )
        return cls(**data)

    def structural_dict(self) -> dict[str, Any]:
        """Return palette/bind facts; per-vertex weights are mutable payload."""

        data = self.to_dict()
        for key in (
            "skin_row_count",
            "influence_count",
            "max_influences",
            "invalid_influence_rows",
            "unnormalized_rows",
            "influence_sha256",
        ):
            data.pop(key, None)
        return data


@dataclass(frozen=True, slots=True)
class HeadDonorSnapshot:
    schema: str
    version: int
    game: str
    resref: str
    resource_view: str
    mdl_sha256: str
    mdx_sha256: str
    model_name: str
    geometry_root_name: str
    attachment_target_name: str
    supermodel: str
    game_version: str
    model_type: int
    detected_character_mode: str
    local_animation_names: tuple[str, ...]
    inherited_node_declaration: int
    local_node_count: int
    retail_bb_min: tuple[float, float, float]
    retail_bb_max: tuple[float, float, float]
    retail_radius: float
    preview_bb_min: tuple[float, float, float]
    preview_bb_max: tuple[float, float, float]
    preview_radius: float
    mutable_payload_node_ordinals: tuple[int, ...]
    nodes: tuple[HeadDonorNodeContract, ...]
    meshes: tuple[HeadDonorMeshPayload, ...]
    skins: tuple[HeadDonorSkinContract, ...]
    structural_sha256: str
    provenance: dict[str, Any] = field(default_factory=dict)
    compatibility: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["nodes"] = [row.to_dict() for row in self.nodes]
        data["meshes"] = [row.to_dict() for row in self.meshes]
        data["skins"] = [row.to_dict() for row in self.skins]
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HeadDonorSnapshot":
        data = dict(payload)
        if data.get("schema") != HEAD_DONOR_CONTRACT_SCHEMA:
            raise ValueError("Unsupported head donor contract schema")
        if int(data.get("version") or 0) != HEAD_DONOR_CONTRACT_VERSION:
            raise ValueError("Unsupported head donor contract version")
        data["local_animation_names"] = tuple(
            str(value) for value in data.get("local_animation_names") or ()
        )
        data["mutable_payload_node_ordinals"] = tuple(
            int(value)
            for value in data.get("mutable_payload_node_ordinals") or ()
        )
        for key in (
            "retail_bb_min",
            "retail_bb_max",
            "preview_bb_min",
            "preview_bb_max",
        ):
            data[key] = tuple(float(value) for value in data.get(key) or ())
        data["nodes"] = tuple(
            HeadDonorNodeContract.from_dict(row)
            for row in data.get("nodes") or ()
        )
        data["meshes"] = tuple(
            HeadDonorMeshPayload.from_dict(row)
            for row in data.get("meshes") or ()
        )
        data["skins"] = tuple(
            HeadDonorSkinContract.from_dict(row)
            for row in data.get("skins") or ()
        )
        data["provenance"] = dict(data.get("provenance") or {})
        data["compatibility"] = dict(data.get("compatibility") or {})
        snapshot = cls(**data)
        expected = _structural_fingerprint(snapshot.structural_dict())
        if snapshot.structural_sha256 != expected:
            raise ValueError(
                "Head donor contract structural fingerprint does not match its content"
            )
        return snapshot

    def structural_dict(self) -> dict[str, Any]:
        """Return only facts an unchanged-DAG transplant must preserve."""

        return {
            "game": self.game,
            "geometry_root_name": self.geometry_root_name,
            "attachment_target_name": self.attachment_target_name,
            "supermodel": self.supermodel,
            "game_version": self.game_version,
            "model_type": self.model_type,
            "detected_character_mode": self.detected_character_mode,
            "local_animation_names": list(self.local_animation_names),
            "inherited_node_declaration": self.inherited_node_declaration,
            "local_node_count": self.local_node_count,
            "retail_bb_min": list(self.retail_bb_min),
            "retail_bb_max": list(self.retail_bb_max),
            "retail_radius": self.retail_radius,
            "mutable_payload_node_ordinals": list(
                self.mutable_payload_node_ordinals
            ),
            "nodes": [row.to_dict() for row in self.nodes],
            "skins": [row.structural_dict() for row in self.skins],
        }


@dataclass(frozen=True, slots=True)
class HeadDonorEligibilityIssue:
    check_id: str
    severity: str
    message: str
    facts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadDonorEligibilityReport:
    issues: tuple[HeadDonorEligibilityIssue, ...] = ()

    @property
    def eligible(self) -> bool:
        return not any(row.severity == "error" for row in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "issues": [row.to_dict() for row in self.issues],
        }


@dataclass(frozen=True, slots=True)
class HeadDonorDifference:
    path: str
    category: str
    expected: Any
    actual: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadDonorContractDiff:
    blocking: tuple[HeadDonorDifference, ...] = ()
    allowed_payload_changes: tuple[HeadDonorDifference, ...] = ()

    @property
    def structurally_compatible(self) -> bool:
        return not self.blocking

    def to_dict(self) -> dict[str, Any]:
        return {
            "structurally_compatible": self.structurally_compatible,
            "blocking": [row.to_dict() for row in self.blocking],
            "allowed_payload_changes": [
                row.to_dict() for row in self.allowed_payload_changes
            ],
        }


def capture_head_donor_snapshot(
    model: Any,
    *,
    game: str,
    resref: str,
    resource_view: str,
    mdl_sha256: str,
    mdx_sha256: str,
    provenance: Mapping[str, Any] | None = None,
    compatibility: Mapping[str, Any] | None = None,
) -> HeadDonorSnapshot:
    """Capture donor facts before any custom geometry mutation."""

    nodes = _model_nodes(model)
    ordinal_by_id = {id(node): ordinal for ordinal, node in enumerate(nodes)}
    node_rows = tuple(
        _capture_node(node, ordinal, ordinal_by_id)
        for ordinal, node in enumerate(nodes)
    )
    mesh_rows = tuple(
        _capture_mesh(node, ordinal)
        for ordinal, node in enumerate(nodes)
        if _is_mesh(node)
    )
    skin_rows = tuple(
        _capture_skin(node, ordinal, ordinal_by_id)
        for ordinal, node in enumerate(nodes)
        if _is_skin(node)
    )
    mutable_payload_ordinals = tuple(
        row.node_ordinal
        for row in skin_rows
        if row.node_name.casefold() == "head"
        and row.parent_ordinal == 0
    )
    root = getattr(model, "root_node", None)
    preview_min, preview_max, preview_radius = _preview_bounds(model)
    data = {
        "schema": HEAD_DONOR_CONTRACT_SCHEMA,
        "version": HEAD_DONOR_CONTRACT_VERSION,
        "game": str(game or "").upper(),
        "resref": str(resref or "").strip(),
        "resource_view": str(getattr(resource_view, "value", resource_view) or ""),
        "mdl_sha256": str(mdl_sha256 or "").lower(),
        "mdx_sha256": str(mdx_sha256 or "").lower(),
        "model_name": str(getattr(model, "name", "") or ""),
        "geometry_root_name": str(getattr(root, "name", "") or ""),
        "attachment_target_name": str(
            getattr(model, "super_root_node_name", "") or ""
        ),
        "supermodel": str(getattr(model, "supermodel", "") or ""),
        "game_version": _game_version_label(model),
        "model_type": _safe_int(getattr(model, "model_type", 0)),
        "detected_character_mode": _detected_character_mode(model),
        "local_animation_names": tuple(
            str(getattr(animation, "name", "") or "")
            for animation in list(getattr(model, "animations", []) or [])
        ),
        "inherited_node_declaration": _safe_int(
            getattr(model, "geometry_node_count", 0)
        ),
        "local_node_count": len(nodes),
        "retail_bb_min": _float_tuple(getattr(model, "bb_min", ()), 3),
        "retail_bb_max": _float_tuple(getattr(model, "bb_max", ()), 3),
        "retail_radius": _safe_float(getattr(model, "radius", 0.0)),
        "preview_bb_min": preview_min,
        "preview_bb_max": preview_max,
        "preview_radius": preview_radius,
        "mutable_payload_node_ordinals": mutable_payload_ordinals,
        "nodes": node_rows,
        "meshes": mesh_rows,
        "skins": skin_rows,
        "structural_sha256": "",
        "provenance": dict(provenance or {}),
        "compatibility": dict(compatibility or {}),
    }
    provisional = HeadDonorSnapshot(**data)
    data["structural_sha256"] = _structural_fingerprint(
        provisional.structural_dict()
    )
    return HeadDonorSnapshot(**data)


def validate_head_donor_snapshot(
    snapshot: HeadDonorSnapshot,
) -> HeadDonorEligibilityReport:
    """Validate that a captured resource is safe to offer as a native donor."""

    issues: list[HeadDonorEligibilityIssue] = []

    def issue(
        check_id: str,
        message: str,
        *,
        severity: str = "error",
        **facts: Any,
    ) -> None:
        issues.append(
            HeadDonorEligibilityIssue(
                check_id=check_id,
                severity=severity,
                message=message,
                facts=facts,
            )
        )

    if not _SHA256_RE.fullmatch(snapshot.mdl_sha256):
        issue(
            "head.donor.source.mdl_hash",
            "The donor MDL does not have a valid SHA-256 identity.",
        )
    if not _SHA256_RE.fullmatch(snapshot.mdx_sha256):
        issue(
            "head.donor.source.mdx_hash",
            "The donor MDX does not have a valid SHA-256 identity.",
        )
    if snapshot.game not in {"K1", "K2"}:
        issue(
            "head.donor.game",
            "The donor must belong to KOTOR I or KOTOR II.",
            game=snapshot.game,
        )
    if snapshot.model_type != 4:
        issue(
            "head.donor.model_type",
            "The donor is not an Odyssey character-class model.",
            model_type=snapshot.model_type,
        )
    if snapshot.detected_character_mode != "head":
        issue(
            "head.donor.character_mode",
            "The selected resource is not classified as a modular head.",
            detected_mode=snapshot.detected_character_mode,
        )
    if snapshot.game_version and snapshot.game_version != snapshot.game:
        issue(
            "head.donor.game_version",
            "The decoded MDL family does not match the selected game.",
            selected_game=snapshot.game,
            decoded_game=snapshot.game_version,
        )
    if snapshot.resource_view not in {"stock_only", "effective_override"}:
        issue(
            "head.donor.resource_view",
            "The donor resource-view policy is unknown.",
            resource_view=snapshot.resource_view,
        )
    if (
        snapshot.resource_view == "stock_only"
        and bool(snapshot.provenance.get("effective_override", False))
    ):
        issue(
            "head.donor.stock_provenance",
            "Stock-only selection resolved to Override bytes.",
        )
    if not snapshot.nodes or not snapshot.geometry_root_name:
        issue(
            "head.donor.geometry_root",
            "The donor has no usable geometry root.",
        )
    elif snapshot.model_name.casefold() != snapshot.geometry_root_name.casefold():
        issue(
            "head.donor.root_identity",
            "The donor model and geometry-root identities do not match.",
            model_name=snapshot.model_name,
            root_name=snapshot.geometry_root_name,
        )
    if snapshot.local_node_count != len(snapshot.nodes):
        issue(
            "head.donor.local_node_count",
            "The captured local-node count does not match the donor DAG.",
            declared=snapshot.local_node_count,
            captured=len(snapshot.nodes),
        )
    if snapshot.inherited_node_declaration < snapshot.local_node_count:
        issue(
            "head.donor.inherited_node_declaration",
            "The inherited node declaration is smaller than the local DAG.",
            declaration=snapshot.inherited_node_declaration,
            local_nodes=snapshot.local_node_count,
        )
    if not snapshot.attachment_target_name:
        issue(
            "head.donor.attachment_target",
            "The modular head has no model-header body attachment target.",
        )
    else:
        matches = [
            row
            for row in snapshot.nodes
            if row.name.casefold()
            == snapshot.attachment_target_name.casefold()
        ]
        if len(matches) != 1:
            issue(
                "head.donor.attachment_target",
                "The body attachment target must resolve to exactly one local node.",
                target=snapshot.attachment_target_name,
                matches=len(matches),
            )
        elif matches[0].ordinal == 0:
            issue(
                "head.donor.root_link_distinction",
                "A modular head's geometry root and body attachment target "
                "must remain distinct.",
            )
    if not snapshot.supermodel or snapshot.supermodel.upper() in {"NULL", "NONE"}:
        issue(
            "head.donor.supermodel",
            "The donor has no inherited animation supermodel.",
        )
    if snapshot.local_animation_names:
        issue(
            "head.donor.local_animations",
            "This donor carries local animation clips; they will be preserved "
            "and require explicit review.",
            severity="warning",
            animations=list(snapshot.local_animation_names),
        )

    sparse = [row.sparse_node_number for row in snapshot.nodes]
    if len(set(sparse)) != len(sparse):
        issue(
            "head.donor.sparse_node_identity",
            "Sparse native node identities are not unique.",
        )
    for row in snapshot.nodes:
        if row.ordinal < 0 or row.ordinal >= len(snapshot.nodes):
            issue(
                "head.donor.node_ordinal",
                "A donor node has an invalid preorder ordinal.",
                node=row.name,
                ordinal=row.ordinal,
            )
        if row.ordinal == 0 and row.parent_ordinal is not None:
            issue(
                "head.donor.root_parent",
                "The geometry root unexpectedly has a parent.",
            )
        if row.parent_ordinal is not None and not (
            0 <= row.parent_ordinal < len(snapshot.nodes)
        ):
            issue(
                "head.donor.parent_identity",
                "A donor node references a parent outside the local DAG.",
                node=row.name,
                parent_ordinal=row.parent_ordinal,
            )
        if (
            row.parent_ordinal is not None
            and row.parent_ordinal >= row.ordinal
        ):
            issue(
                "head.donor.dag_preorder",
                "A donor parent does not precede its child in native preorder.",
                node=row.name,
                ordinal=row.ordinal,
                parent_ordinal=row.parent_ordinal,
            )
        for child_ordinal in row.child_ordinals:
            if not 0 <= child_ordinal < len(snapshot.nodes):
                issue(
                    "head.donor.child_identity",
                    "A donor node references a child outside the local DAG.",
                    node=row.name,
                    child_ordinal=child_ordinal,
                )
            elif snapshot.nodes[child_ordinal].parent_ordinal != row.ordinal:
                issue(
                    "head.donor.parent_child_reciprocity",
                    "The donor parent/child identities are inconsistent.",
                    node=row.name,
                    child_ordinal=child_ordinal,
                )
        if not _all_finite(
            (*row.position, *row.rotation, *row.bb_min, *row.bb_max, row.radius)
        ):
            issue(
                "head.donor.node_finite",
                "A donor node contains non-finite transform or bound values.",
                node=row.name,
            )
    if not _all_finite(
        (
            *snapshot.retail_bb_min,
            *snapshot.retail_bb_max,
            snapshot.retail_radius,
            *snapshot.preview_bb_min,
            *snapshot.preview_bb_max,
            snapshot.preview_radius,
        )
    ):
        issue(
            "head.donor.bounds_finite",
            "The donor contains non-finite retail or preview bounds.",
        )

    primary_skins = [
        row
        for row in snapshot.skins
        if row.node_name.casefold() == "head"
        and row.parent_ordinal == 0
    ]
    if len(primary_skins) != 1:
        issue(
            "head.donor.primary_skin",
            "A modular head donor must expose exactly one direct-root skin "
            "named 'head'.",
            matches=len(primary_skins),
        )
    if tuple(row.node_ordinal for row in primary_skins) != (
        snapshot.mutable_payload_node_ordinals
    ):
        issue(
            "head.donor.mutable_payload_identity",
            "The mutable rendered payload identity does not match the "
            "direct-root head skin.",
            expected=[row.node_ordinal for row in primary_skins],
            captured=list(snapshot.mutable_payload_node_ordinals),
        )
    for skin in snapshot.skins:
        palette_size = len(skin.bone_palette)
        if palette_size > 16:
            issue(
                "head.donor.palette_limit",
                "The donor skin palette exceeds Odyssey's 16-slot limit.",
                node=skin.node_name,
                slots=palette_size,
            )
        if len(skin.bone_node_indices) != palette_size:
            issue(
                "head.donor.palette_targets",
                "The donor palette names and node identities have different lengths.",
                node=skin.node_name,
                palette=palette_size,
                targets=len(skin.bone_node_indices),
            )
        if len(skin.qbone_rows) != snapshot.local_node_count:
            issue(
                "head.donor.qbone_rows",
                "qBone rows must use the donor's local-node convention.",
                node=skin.node_name,
                rows=len(skin.qbone_rows),
                local_nodes=snapshot.local_node_count,
            )
        if len(skin.tbone_rows) != snapshot.local_node_count:
            issue(
                "head.donor.tbone_rows",
                "tBone rows must use the donor's local-node convention.",
                node=skin.node_name,
                rows=len(skin.tbone_rows),
                local_nodes=snapshot.local_node_count,
            )
        if len(skin.bone_map_floats) not in {
            0,
            snapshot.local_node_count,
        }:
            issue(
                "head.donor.bone_map_layout",
                "The bone-map float layout is neither absent nor local-node indexed.",
                node=skin.node_name,
                rows=len(skin.bone_map_floats),
                local_nodes=snapshot.local_node_count,
            )
        if skin.invalid_influence_rows:
            issue(
                "head.donor.influences_finite",
                "The donor contains invalid, negative, or over-limit skin rows.",
                node=skin.node_name,
                invalid_rows=skin.invalid_influence_rows,
            )
        if skin.unnormalized_rows:
            issue(
                "head.donor.influences_normalized",
                "The donor contains unnormalized skin rows.",
                node=skin.node_name,
                unnormalized_rows=skin.unnormalized_rows,
            )
        if skin in primary_skins:
            palette_names = {
                name.casefold() for name in skin.bone_palette if name
            }
            if "head_g" not in palette_names:
                issue(
                    "head.donor.head_influence",
                    "The donor skin palette has no head_g fallback influence.",
                    node=skin.node_name,
                )
            if snapshot.attachment_target_name.casefold() not in palette_names:
                issue(
                    "head.donor.attachment_influence",
                    "The donor skin palette does not target its attachment node.",
                    node=skin.node_name,
                    target=snapshot.attachment_target_name,
                )

    return HeadDonorEligibilityReport(tuple(issues))


def compare_head_donor_contract(
    snapshot: HeadDonorSnapshot,
    model: Any,
    *,
    output_resref: str = "",
    allowed_local_animation_names: Iterable[str] = (),
) -> HeadDonorContractDiff:
    """Compare an edited or reloaded model to its immutable donor contract.

    Geometry, UV, material, and per-vertex skin rows may change on the primary
    head skin and on any existing render nodes explicitly certified by the
    saved component recipe.  DAG, native identities, palette/bind arrays, raw
    bounds, inherited declaration, attachment link, and supermodel may not.
    Local animation additions require an explicit allow-list; removals and
    unlisted additions remain structural failures.
    """

    current = capture_head_donor_snapshot(
        model,
        game=snapshot.game,
        resref=snapshot.resref,
        resource_view=snapshot.resource_view,
        mdl_sha256=snapshot.mdl_sha256,
        mdx_sha256=snapshot.mdx_sha256,
    )
    blocking: list[HeadDonorDifference] = []
    allowed: list[HeadDonorDifference] = []
    allowed_root = str(output_resref or "").strip()

    def blocked(path: str, expected: Any, actual: Any, message: str) -> None:
        blocking.append(
            HeadDonorDifference(
                path=path,
                category="structural",
                expected=expected,
                actual=actual,
                message=message,
            )
        )

    def payload(path: str, expected: Any, actual: Any, message: str) -> None:
        allowed.append(
            HeadDonorDifference(
                path=path,
                category="render_payload",
                expected=expected,
                actual=actual,
                message=message,
            )
        )

    extra_payload_ordinals: list[int] = []
    for raw in list(
        dict(snapshot.compatibility or {}).get(
            "component_payload_node_ordinals",
            (),
        )
        or ()
    ):
        try:
            extra_payload_ordinals.append(int(raw))
        except (TypeError, ValueError):
            blocked(
                "compatibility.component_payload_node_ordinals",
                "integer node ordinals",
                raw,
                "The component recipe contains a non-integer payload node identity.",
            )
    mesh_ordinals = {row.node_ordinal for row in snapshot.meshes}
    for ordinal in sorted(set(extra_payload_ordinals)):
        if ordinal not in mesh_ordinals:
            blocked(
                "compatibility.component_payload_node_ordinals",
                sorted(mesh_ordinals),
                ordinal,
                "The component recipe references a non-mesh donor node.",
            )
    allowed_payload_ordinals = set(
        snapshot.mutable_payload_node_ordinals
    ) | (set(extra_payload_ordinals) & mesh_ordinals)
    disabled_render_ordinals: set[int] = set()
    for raw in list(
        dict(snapshot.compatibility or {}).get(
            "disabled_render_node_ordinals",
            (),
        )
        or ()
    ):
        try:
            ordinal = int(raw)
        except (TypeError, ValueError):
            blocked(
                "compatibility.disabled_render_node_ordinals",
                "integer node ordinals",
                raw,
                "The facial recipe contains a non-integer retired-node identity.",
            )
            continue
        if ordinal not in allowed_payload_ordinals:
            blocked(
                "compatibility.disabled_render_node_ordinals",
                sorted(allowed_payload_ordinals),
                ordinal,
                "A retired render node must also be an allowed payload node.",
            )
            continue
        disabled_render_ordinals.add(ordinal)

    for name in (
        "game_version",
        "model_type",
        "detected_character_mode",
        "attachment_target_name",
        "supermodel",
        "inherited_node_declaration",
        "local_node_count",
        "retail_bb_min",
        "retail_bb_max",
        "retail_radius",
        "mutable_payload_node_ordinals",
    ):
        expected = getattr(snapshot, name)
        actual = getattr(current, name)
        if not _values_equal(expected, actual):
            blocked(
                name,
                expected,
                actual,
                f"Donor structural field changed: {name}.",
            )

    allowed_animation_names = tuple(
        str(value or "").strip()
        for value in allowed_local_animation_names
        if str(value or "").strip()
    )
    expected_animation_names = list(snapshot.local_animation_names)
    expected_animation_keys = {
        value.casefold() for value in expected_animation_names
    }
    for value in allowed_animation_names:
        if value.casefold() not in expected_animation_keys:
            expected_animation_names.append(value)
            expected_animation_keys.add(value.casefold())
    actual_animation_names = list(current.local_animation_names)
    if [value.casefold() for value in actual_animation_names] != [
        value.casefold() for value in expected_animation_names
    ]:
        blocked(
            "local_animation_names",
            tuple(expected_animation_names),
            tuple(actual_animation_names),
            "Local animation inventory contains a removal or unlisted addition.",
        )
    elif tuple(actual_animation_names) != snapshot.local_animation_names:
        allowed.append(
            HeadDonorDifference(
                path="local_animation_names",
                category="animation_payload",
                expected=snapshot.local_animation_names,
                actual=tuple(actual_animation_names),
                message="Explicitly allowed local animation override was added.",
            )
        )

    for name in ("model_name", "geometry_root_name"):
        expected = getattr(snapshot, name)
        actual = getattr(current, name)
        if _values_equal(expected, actual):
            continue
        if allowed_root and str(actual).casefold() == allowed_root.casefold():
            payload(
                name,
                expected,
                actual,
                "The geometry root identity was renamed to the requested output resref.",
            )
        else:
            blocked(
                name,
                expected,
                actual,
                "The geometry root may only change to the explicit output resref.",
            )

    if len(snapshot.nodes) != len(current.nodes):
        blocked(
            "nodes",
            len(snapshot.nodes),
            len(current.nodes),
            "The donor DAG node count changed.",
        )
    for ordinal, (expected_node, actual_node) in enumerate(
        zip(snapshot.nodes, current.nodes)
    ):
        for name in (
            "parent_ordinal",
            "child_ordinals",
            "flags",
            "type_label",
            "dense_name_index",
            "sparse_node_number",
            "position",
            "rotation",
            "is_mesh",
            "is_skin",
            "socket_category",
            "bb_min",
            "bb_max",
            "radius",
        ):
            expected = getattr(expected_node, name)
            actual = getattr(actual_node, name)
            if not _values_equal(expected, actual):
                blocked(
                    f"nodes[{ordinal}].{name}",
                    expected,
                    actual,
                    f"Native donor node contract changed at ordinal {ordinal}.",
                )
        if not _values_equal(expected_node.render, actual_node.render):
            if ordinal in disabled_render_ordinals and not actual_node.render:
                payload(
                    f"nodes[{ordinal}].render",
                    expected_node.render,
                    actual_node.render,
                    "A visible donor mesh replaced by complete custom facial art was retired.",
                )
            else:
                blocked(
                    f"nodes[{ordinal}].render",
                    expected_node.render,
                    actual_node.render,
                    f"Native donor node contract changed at ordinal {ordinal}.",
                )
        if expected_node.name != actual_node.name:
            root_rename = (
                ordinal == 0
                and allowed_root
                and actual_node.name.casefold() == allowed_root.casefold()
            )
            if root_rename:
                payload(
                    f"nodes[{ordinal}].name",
                    expected_node.name,
                    actual_node.name,
                    "The geometry-root node was renamed to the output resref.",
                )
            else:
                blocked(
                    f"nodes[{ordinal}].name",
                    expected_node.name,
                    actual_node.name,
                    "A native donor node name changed.",
                )

    if len(snapshot.skins) != len(current.skins):
        blocked(
            "skins",
            len(snapshot.skins),
            len(current.skins),
            "The number of donor skin nodes changed.",
        )
    for ordinal, (expected_skin, actual_skin) in enumerate(
        zip(snapshot.skins, current.skins)
    ):
        expected_structural = expected_skin.structural_dict()
        actual_structural = actual_skin.structural_dict()
        for name in expected_structural:
            if not _values_equal(
                expected_structural[name],
                actual_structural.get(name),
            ):
                blocked(
                    f"skins[{ordinal}].{name}",
                    expected_structural[name],
                    actual_structural.get(name),
                    "The donor palette or bind-array contract changed.",
                )
        for name in (
            "skin_row_count",
            "influence_count",
            "max_influences",
            "invalid_influence_rows",
            "unnormalized_rows",
            "influence_sha256",
        ):
            expected = getattr(expected_skin, name)
            actual = getattr(actual_skin, name)
            if not _values_equal(expected, actual):
                if (
                    expected_skin.node_ordinal
                    in allowed_payload_ordinals
                ):
                    payload(
                        f"skins[{ordinal}].{name}",
                        expected,
                        actual,
                        "Per-vertex skin payload changed.",
                    )
                else:
                    blocked(
                        f"skins[{ordinal}].{name}",
                        expected,
                        actual,
                        "Skin payload changed outside the selected rendered head skin.",
                    )

    expected_meshes = {row.node_ordinal: row for row in snapshot.meshes}
    actual_meshes = {row.node_ordinal: row for row in current.meshes}
    for node_ordinal in sorted(set(expected_meshes) | set(actual_meshes)):
        expected = expected_meshes.get(node_ordinal)
        actual = actual_meshes.get(node_ordinal)
        if expected != actual:
            if node_ordinal in allowed_payload_ordinals:
                payload(
                    f"meshes[{node_ordinal}]",
                    expected.to_dict() if expected else None,
                    actual.to_dict() if actual else None,
                    "Selected rendered head geometry, UV, or material payload changed.",
                )
            else:
                blocked(
                    f"meshes[{node_ordinal}]",
                    expected.to_dict() if expected else None,
                    actual.to_dict() if actual else None,
                    "Geometry changed outside the selected rendered head payload.",
                )
    for name in ("preview_bb_min", "preview_bb_max", "preview_radius"):
        expected = getattr(snapshot, name)
        actual = getattr(current, name)
        if not _values_equal(expected, actual):
            payload(
                name,
                expected,
                actual,
                "Editor-only preview bounds changed with rendered geometry.",
            )
    return HeadDonorContractDiff(tuple(blocking), tuple(allowed))


def _capture_node(
    node: Any,
    ordinal: int,
    ordinal_by_id: Mapping[int, int],
) -> HeadDonorNodeContract:
    parent = getattr(node, "parent", None)
    children = list(getattr(node, "children", []) or [])
    return HeadDonorNodeContract(
        ordinal=ordinal,
        name=str(getattr(node, "name", "") or ""),
        parent_ordinal=(
            ordinal_by_id.get(id(parent)) if parent is not None else None
        ),
        child_ordinals=tuple(
            ordinal_by_id[id(child)]
            for child in children
            if id(child) in ordinal_by_id
        ),
        flags=_safe_int(getattr(node, "flags", 0)),
        type_label=str(getattr(node, "type_label", "") or ""),
        dense_name_index=_safe_int(getattr(node, "index", -1), -1),
        sparse_node_number=_safe_int(getattr(node, "number", -1), -1),
        position=_float_tuple(getattr(node, "position", ()), 3),
        rotation=_float_tuple(
            getattr(node, "rotation", ()),
            4,
            last_default=1.0,
        ),
        is_mesh=_is_mesh(node),
        is_skin=_is_skin(node),
        render=bool(getattr(node, "render", True)),
        socket_category=_socket_category(
            str(getattr(node, "name", "") or "")
        ),
        bb_min=_float_tuple(getattr(node, "bb_min", ()), 3),
        bb_max=_float_tuple(getattr(node, "bb_max", ()), 3),
        radius=_safe_float(getattr(node, "radius", 0.0)),
    )


def _capture_mesh(node: Any, ordinal: int) -> HeadDonorMeshPayload:
    geometry = {
        "vertices": _json_rows(getattr(node, "vertices", ())),
        "faces": _json_rows(getattr(node, "faces", ())),
        "normals": _json_rows(getattr(node, "normals", ())),
        "uvs": _json_rows(getattr(node, "uvs", ())),
        "uvs_lm": _json_rows(getattr(node, "uvs_lm", ())),
    }
    return HeadDonorMeshPayload(
        node_ordinal=ordinal,
        node_name=str(getattr(node, "name", "") or ""),
        vertex_count=len(list(getattr(node, "vertices", []) or [])),
        face_count=len(list(getattr(node, "faces", []) or [])),
        normal_count=len(list(getattr(node, "normals", []) or [])),
        uv_count=len(list(getattr(node, "uvs", []) or [])),
        uv2_count=len(list(getattr(node, "uvs_lm", []) or [])),
        texture=str(getattr(node, "texture", "") or ""),
        lightmap=str(getattr(node, "lightmap", "") or ""),
        geometry_sha256=_hash_json(geometry),
    )


def _capture_skin(
    node: Any,
    ordinal: int,
    ordinal_by_id: Mapping[int, int],
) -> HeadDonorSkinContract:
    parent = getattr(node, "parent", None)
    rows = list(getattr(node, "skin_data", []) or [])
    influence_rows: list[list[list[float | int]]] = []
    influence_count = 0
    max_influences = 0
    invalid_rows = 0
    unnormalized_rows = 0
    for row in rows:
        influences = list(getattr(row, "influences", []) or [])
        encoded_row: list[list[float | int]] = []
        valid = len(influences) <= 4
        total = 0.0
        for influence in influences:
            bone_index = _safe_int(getattr(influence, "bone_index", -1), -1)
            weight = _safe_float(getattr(influence, "weight", math.nan), math.nan)
            encoded_row.append([bone_index, weight])
            influence_count += 1
            total += weight
            if (
                bone_index < 0
                or not math.isfinite(weight)
                or weight < 0.0
            ):
                valid = False
        if not valid:
            invalid_rows += 1
        if influences and (
            not math.isfinite(total)
            or abs(total - 1.0) > _WEIGHT_TOLERANCE
        ):
            unnormalized_rows += 1
        max_influences = max(max_influences, len(influences))
        influence_rows.append(encoded_row)
    qbone_rows = tuple(
        _float_tuple(row, 4, last_default=1.0)
        for row in list(getattr(node, "qbone_list", []) or [])
    )
    tbone_rows = tuple(
        _float_tuple(row, 3)
        for row in list(getattr(node, "tbone_list", []) or [])
    )
    return HeadDonorSkinContract(
        node_ordinal=ordinal,
        node_name=str(getattr(node, "name", "") or ""),
        parent_ordinal=(
            ordinal_by_id.get(id(parent)) if parent is not None else None
        ),
        bone_palette=tuple(
            str(value)
            for value in list(getattr(node, "bone_map", []) or [])
        ),
        bone_node_indices=tuple(
            _safe_int(value, -1)
            for value in list(
                getattr(node, "bone_node_indices", []) or []
            )
        ),
        bone_map_floats=tuple(
            _safe_float(value)
            for value in list(getattr(node, "bone_map_floats", []) or [])
        ),
        qbone_rows=qbone_rows,
        tbone_rows=tbone_rows,
        bind_row_count=max(len(qbone_rows), len(tbone_rows)),
        skin_row_count=len(rows),
        influence_count=influence_count,
        max_influences=max_influences,
        invalid_influence_rows=invalid_rows,
        unnormalized_rows=unnormalized_rows,
        influence_sha256=_hash_json(influence_rows),
    )


def _model_nodes(model: Any) -> list[Any]:
    all_nodes = getattr(model, "all_nodes", None)
    if callable(all_nodes):
        return list(all_nodes())
    root = getattr(model, "root_node", None)
    if root is None:
        return []
    result: list[Any] = []
    stack = [root]
    visited: set[int] = set()
    while stack:
        node = stack.pop()
        identity = id(node)
        if identity in visited:
            continue
        visited.add(identity)
        result.append(node)
        stack.extend(reversed(list(getattr(node, "children", []) or [])))
    return result


def _preview_bounds(
    model: Any,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]:
    bounds = getattr(model, "_gr_render_bounds", None)
    if (
        isinstance(bounds, (tuple, list))
        and len(bounds) == 2
    ):
        minimum = _float_tuple(bounds[0], 3)
        maximum = _float_tuple(bounds[1], 3)
    else:
        minimum = _float_tuple(getattr(model, "bb_min", ()), 3)
        maximum = _float_tuple(getattr(model, "bb_max", ()), 3)
    radius = _safe_float(
        getattr(model, "_gr_render_radius", getattr(model, "radius", 0.0))
    )
    return minimum, maximum, radius


def _is_mesh(node: Any) -> bool:
    return bool(getattr(node, "is_mesh", False))


def _is_skin(node: Any) -> bool:
    return bool(getattr(node, "is_skin", False))


def _socket_category(name: str) -> str:
    lowered = str(name or "").strip().casefold()
    if lowered in _HEAD_SOCKET_CATEGORIES:
        return _HEAD_SOCKET_CATEGORIES[lowered]
    if lowered.startswith("f_"):
        return "facial_control"
    if lowered.endswith("hook"):
        return "hook"
    return ""


def _game_version_label(model: Any) -> str:
    value = getattr(model, "game_version", "")
    name = str(getattr(value, "name", "") or "")
    if name:
        return name.upper()
    text = str(value or "").upper()
    if text in {"1", "GAMEVERSION.K1"}:
        return "K1"
    if text in {"2", "GAMEVERSION.K2"}:
        return "K2"
    return text


def _detected_character_mode(model: Any) -> str:
    try:
        from src.core.geometry.model_data import detect_character_mode

        mode = detect_character_mode(model)
        return str(getattr(mode, "value", mode) or "")
    except Exception:
        return ""


def _json_rows(values: Iterable[Any]) -> list[list[float | int]]:
    rows: list[list[float | int]] = []
    for value in list(values or []):
        try:
            rows.append(
                [
                    int(item)
                    if isinstance(item, int) and not isinstance(item, bool)
                    else float(item)
                    for item in value
                ]
            )
        except (TypeError, ValueError):
            rows.append([])
    return rows


def _float_tuple(
    values: Any,
    count: int,
    *,
    last_default: float | None = None,
) -> tuple[float, ...]:
    raw = list(values or ())
    result: list[float] = []
    for index in range(count):
        default = (
            last_default
            if last_default is not None and index == count - 1
            else 0.0
        )
        try:
            result.append(float(raw[index]))
        except (IndexError, TypeError, ValueError):
            result.append(float(default))
    return tuple(result)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _all_finite(values: Iterable[Any]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _values_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return math.isclose(
            float(left),
            float(right),
            rel_tol=0.0,
            abs_tol=_FLOAT_TOLERANCE,
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _values_equal(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_equal(left[key], right[key]) for key in left
        )
    return left == right


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child)
            for key, child in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "<nan>"
        return "<positive-infinity>" if value > 0 else "<negative-infinity>"
    return value


def _structural_fingerprint(value: Mapping[str, Any]) -> str:
    return _hash_json(value)


__all__ = [
    "HEAD_DONOR_CONTRACT_SCHEMA",
    "HEAD_DONOR_CONTRACT_VERSION",
    "HeadDonorContractDiff",
    "HeadDonorDifference",
    "HeadDonorEligibilityIssue",
    "HeadDonorEligibilityReport",
    "HeadDonorMeshPayload",
    "HeadDonorNodeContract",
    "HeadDonorSkinContract",
    "HeadDonorSnapshot",
    "capture_head_donor_snapshot",
    "compare_head_donor_contract",
    "validate_head_donor_snapshot",
]
