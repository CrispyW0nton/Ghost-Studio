"""Build a donor-preserving facial head from semantic custom-art anatomy.

The ordinary head skin transplant intentionally merges all selected art into
one native skin. Facial-performance heads need a stricter partition:

* face shell -> donor ``head`` skin with barycentric retail weights;
* disconnected accessories -> the same skin, rigid to ``head_g``;
* upper teeth -> native upper-teeth node under ``head_g``;
* lower teeth and tongue -> native nodes under ``f_jaw_g``;
* eyes and eyelids -> their native carrier nodes.

The native hierarchy, node identities, skin palette, inverse binds,
supermodel, and inherited animation ownership remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.core.characters.head_art_anatomy import (
    HeadArtAnatomyReport,
    HeadArtAnatomyRole,
    component_mesh,
)
from src.core.characters.head_donor_snapshot import (
    HeadDonorSnapshot,
    compare_head_donor_contract,
)
from src.io.head_art_importer import HeadArtPart
from src.math.head_alignment import (
    HeadAlignmentAnchor,
    HeadAlignmentRequest,
    HeadAlignmentResult,
    solve_headhook_alignment,
    transform_point,
    transform_vector,
)
from src.math.head_component_transform import (
    build_head_component_rebase,
    rebase_head_component_channels,
)
from src.math.head_skin_transfer import (
    HeadSkinTransferReport,
    transfer_head_skin_weights,
)


class HeadFacialTransplantError(RuntimeError):
    """Raised when semantic facial art cannot preserve the donor contract."""


@dataclass(frozen=True, slots=True)
class HeadFacialTransplantReport:
    output_resref: str
    texture_resref: str
    anatomy: HeadArtAnatomyReport
    alignment: HeadAlignmentResult
    skin_transfer: HeadSkinTransferReport
    facial_skin_vertex_count: int
    facial_skin_face_count: int
    rigid_accessory_vertex_count: int
    mutable_node_ordinal: int
    mutable_node_name: str
    payload_node_ordinals: tuple[int, ...]
    texture_node_ordinals: tuple[int, ...]
    component_nodes: tuple[tuple[str, str, int, int], ...]
    disabled_donor_render_nodes: tuple[str, ...]
    disabled_donor_render_node_ordinals: tuple[int, ...]
    blocking_difference_paths: tuple[str, ...]
    allowed_difference_paths: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return (
            self.anatomy.ok
            and self.skin_transfer.accepted
            and self.facial_skin_vertex_count > 0
            and self.facial_skin_face_count > 0
            and len(self.component_nodes) == 7
            and not self.blocking_difference_paths
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ghostrigger.head_facial_transplant",
            "version": 1,
            "accepted": self.accepted,
            "output_resref": self.output_resref,
            "texture_resref": self.texture_resref,
            "anatomy": self.anatomy.to_dict(),
            "alignment": self.alignment.to_dict(),
            "skin_transfer": self.skin_transfer.to_dict(),
            "facial_skin_vertex_count": self.facial_skin_vertex_count,
            "facial_skin_face_count": self.facial_skin_face_count,
            "rigid_accessory_vertex_count": self.rigid_accessory_vertex_count,
            "mutable_node_ordinal": self.mutable_node_ordinal,
            "mutable_node_name": self.mutable_node_name,
            "payload_node_ordinals": list(self.payload_node_ordinals),
            "texture_node_ordinals": list(self.texture_node_ordinals),
            "disabled_donor_render_nodes": list(
                self.disabled_donor_render_nodes
            ),
            "disabled_donor_render_node_ordinals": list(
                self.disabled_donor_render_node_ordinals
            ),
            "blocking_difference_paths": list(
                self.blocking_difference_paths
            ),
            "allowed_difference_paths": list(
                self.allowed_difference_paths
            ),
            "component_nodes": [
                {
                    "role": role,
                    "node": node,
                    "vertex_count": vertices,
                    "face_count": faces,
                }
                for role, node, vertices, faces in self.component_nodes
            ],
        }


@dataclass(frozen=True, slots=True)
class HeadFacialTransplantResult:
    model: Any
    donor_snapshot: HeadDonorSnapshot
    report: HeadFacialTransplantReport


ROLE_NODE_NAMES: Mapping[HeadArtAnatomyRole, tuple[str, ...]] = {
    HeadArtAnatomyRole.LEFT_EYE: ("eyeLA",),
    HeadArtAnatomyRole.RIGHT_EYE: ("eyeRA",),
    HeadArtAnatomyRole.LEFT_EYELID: ("eyeLlid",),
    HeadArtAnatomyRole.RIGHT_EYELID: ("eyeRlid",),
    HeadArtAnatomyRole.UPPER_TEETH: (
        "Teethup",
        "teethU",
        "teethUa",
        "teethupper",
    ),
    HeadArtAnatomyRole.LOWER_TEETH: (
        "Teethlow",
        "teethL",
        "teethLa",
        "teethlower",
    ),
    HeadArtAnatomyRole.TONGUE: ("tongue",),
}


def build_head_facial_transplant(
    *,
    donor_model: Any,
    donor_snapshot: HeadDonorSnapshot,
    art_part: HeadArtPart,
    anatomy: HeadArtAnatomyReport,
    output_resref: str,
    texture_resref: str,
    maximum_surface_distance: float = 0.05,
) -> HeadFacialTransplantResult:
    """Build a semantic facial candidate without changing donor hierarchy."""

    if donor_model is None:
        raise HeadFacialTransplantError("A native modular-head donor is required")
    if not isinstance(donor_snapshot, HeadDonorSnapshot):
        raise TypeError("donor_snapshot must be HeadDonorSnapshot")
    if not isinstance(art_part, HeadArtPart):
        raise TypeError("art_part must be HeadArtPart")
    if not isinstance(anatomy, HeadArtAnatomyReport) or not anatomy.ok:
        raise HeadFacialTransplantError(
            "Custom art has not passed semantic facial-anatomy discovery"
        )
    output_name = str(output_resref or "").strip()
    texture_name = str(texture_resref or "").strip()
    if not output_name or len(output_name) > 16:
        raise HeadFacialTransplantError("Output ResRef must contain 1-16 characters")
    if not texture_name or len(texture_name) > 16:
        raise HeadFacialTransplantError("Texture ResRef must contain 1-16 characters")

    donor_nodes = _model_nodes(donor_model)
    donor_skin = _unique_node(
        donor_nodes,
        ("head",),
        require_skin=True,
    )
    palette = tuple(str(value) for value in donor_skin.bone_map)
    head_slot = _palette_slot(palette, "head_g")
    alignment = align_head_art_anatomy_to_donor(
        art_part=art_part,
        anatomy=anatomy,
        donor_model=donor_model,
    )

    shell = anatomy.component(HeadArtAnatomyRole.FACE_SHELL)
    accessories = tuple(
        row
        for row in anatomy.components
        if row.role is HeadArtAnatomyRole.ACCESSORY
    )
    (
        skin_vertices,
        skin_faces,
        skin_normals,
        skin_uvs,
        source_indices,
    ) = component_mesh(art_part, (shell, *accessories))
    transformed_vertices = tuple(
        transform_point(alignment.imported_to_headhook, row)
        for row in skin_vertices
    )
    transformed_normals = tuple(
        transform_vector(
            alignment.imported_to_headhook,
            row,
            normalize=True,
        )
        for row in skin_normals
    )
    accessory_source_indices = {
        value
        for component in accessories
        for value in component.vertex_indices
    }
    rigid_indices = tuple(
        index
        for index, source_index in enumerate(source_indices)
        if source_index in accessory_source_indices
    )
    candidate = deepcopy(donor_model)
    candidate_nodes = _model_nodes(candidate)
    candidate_skin = _unique_node(
        candidate_nodes,
        ("head",),
        require_skin=True,
    )
    (
        donor_transfer_vertices,
        payload_vertices,
        payload_normals,
    ) = _skin_transfer_and_payload_channels(
        donor_skin=donor_skin,
        payload_skin=candidate_skin,
        target_world_vertices=transformed_vertices,
        target_world_normals=transformed_normals,
    )
    transfer = transfer_head_skin_weights(
        donor_vertices=donor_transfer_vertices,
        donor_faces=tuple(donor_skin.faces),
        donor_weight_rows=tuple(donor_skin.skin_data),
        target_vertices=transformed_vertices,
        palette_size=len(palette),
        rigid_fallback_slot=head_slot,
        rigid_target_indices=rigid_indices,
        maximum_surface_distance=float(maximum_surface_distance),
        allow_distance_fallback=True,
    )
    _apply_skin_payload(
        candidate_skin,
        vertices=payload_vertices,
        faces=skin_faces,
        normals=payload_normals,
        uvs=skin_uvs,
        rows=transfer.rows,
        texture_resref=texture_name,
    )

    assigned_node_ids = {id(candidate_skin)}
    component_rows: list[tuple[str, str, int, int]] = []
    for role, node_names in ROLE_NODE_NAMES.items():
        component = anatomy.component(role)
        target_node = _unique_node(candidate_nodes, node_names)
        assigned_node_ids.add(id(target_node))
        vertices, faces, normals, uvs, _source = component_mesh(
            art_part,
            (component,),
        )
        world_vertices = tuple(
            transform_point(alignment.imported_to_headhook, row)
            for row in vertices
        )
        world_normals = tuple(
            transform_vector(
                alignment.imported_to_headhook,
                row,
                normalize=True,
            )
            for row in normals
        )
        local_vertices, local_normals = _world_to_node_channels(
            target_node,
            world_vertices,
            world_normals,
        )
        _apply_rigid_payload(
            target_node,
            vertices=local_vertices,
            faces=faces,
            normals=local_normals,
            uvs=uvs,
            texture_resref=texture_name,
            skin_bone_name=(
                "f_jaw_g"
                if bool(getattr(target_node, "skin_data", ()) or ())
                else ""
            ),
        )
        component_rows.append(
            (role.value, str(target_node.name), len(vertices), len(faces))
        )

    disabled_donor_nodes = _disable_replaced_donor_render_nodes(
        candidate_nodes,
        assigned_node_ids=assigned_node_ids,
    )
    candidate.name = output_name
    if getattr(candidate, "root_node", None) is not None:
        candidate.root_node.name = output_name
    candidate.disable_fog = True
    metadata = getattr(candidate, "metadata", None)
    if not isinstance(metadata, dict):
        metadata = {}
        candidate.metadata = metadata
    mutable_ordinal = candidate_nodes.index(candidate_skin)
    texture_ordinals = tuple(
        index
        for index, node in enumerate(candidate_nodes)
        if id(node) in assigned_node_ids
    )
    disabled_ordinals = tuple(
        index
        for index, node in enumerate(candidate_nodes)
        if str(getattr(node, "name", "") or "") in disabled_donor_nodes
    )
    payload_ordinals = tuple(
        sorted(set(texture_ordinals) | set(disabled_ordinals))
    )
    compatibility = deepcopy(dict(donor_snapshot.compatibility or {}))
    compatibility["component_payload_node_ordinals"] = list(
        payload_ordinals
    )
    compatibility["facial_texture_node_ordinals"] = list(
        texture_ordinals
    )
    compatibility["disabled_render_node_ordinals"] = list(
        disabled_ordinals
    )
    derived_snapshot = replace(
        donor_snapshot,
        compatibility=compatibility,
    )
    diff = compare_head_donor_contract(
        derived_snapshot,
        candidate,
        output_resref=output_name,
    )
    report = HeadFacialTransplantReport(
        output_resref=output_name,
        texture_resref=texture_name,
        anatomy=anatomy,
        alignment=alignment,
        skin_transfer=transfer.report,
        facial_skin_vertex_count=len(transformed_vertices),
        facial_skin_face_count=len(skin_faces),
        rigid_accessory_vertex_count=len(rigid_indices),
        mutable_node_ordinal=mutable_ordinal,
        mutable_node_name=str(candidate_skin.name),
        payload_node_ordinals=payload_ordinals,
        texture_node_ordinals=texture_ordinals,
        component_nodes=tuple(component_rows),
        disabled_donor_render_nodes=disabled_donor_nodes,
        disabled_donor_render_node_ordinals=disabled_ordinals,
        blocking_difference_paths=tuple(
            row.path for row in diff.blocking
        ),
        allowed_difference_paths=tuple(
            row.path for row in diff.allowed_payload_changes
        ),
    )
    metadata["head_facial_transplant"] = report.to_dict()
    if not report.accepted:
        raise HeadFacialTransplantError(
            "Semantic facial transplant did not pass its output contract"
            + (
                ": " + ", ".join(report.blocking_difference_paths[:8])
                if report.blocking_difference_paths
                else ""
            )
        )
    return HeadFacialTransplantResult(
        model=candidate,
        donor_snapshot=derived_snapshot,
        report=report,
    )


def _disable_replaced_donor_render_nodes(
    nodes: Sequence[Any],
    *,
    assigned_node_ids: set[int],
) -> tuple[str, ...]:
    """Retire donor-visible meshes replaced by complete custom head art.

    Facial animation still needs the donor's exact node hierarchy and
    controllers, including helper nodes that carry diagnostic cube geometry.
    Only nodes that were visibly rendered by the donor are retired; assigned
    custom face, eye, lid, teeth, and tongue nodes remain active.
    """

    disabled: list[str] = []
    for node in nodes:
        if id(node) in assigned_node_ids:
            continue
        if not bool(getattr(node, "render", False)):
            continue
        if not (getattr(node, "faces", ()) or getattr(node, "vertices", ())):
            continue
        node.render = False
        _set_static_mesh_alpha(node, 0.0)
        disabled.append(str(getattr(node, "name", "") or "<unnamed>"))
    return tuple(disabled)


def _set_static_mesh_alpha(node: Any, value: float) -> None:
    """Keep the convenience alpha field and its serialized controller aligned."""

    alpha = float(value)
    node.alpha = alpha
    controllers = list(getattr(node, "controllers", ()) or ())
    controller = next(
        (
            row
            for row in controllers
            if int(row.get("type", -1)) in {128, 132}
            or str(row.get("name", "") or "").casefold() == "alpha"
        ),
        None,
    )
    if controller is None:
        controller = {
            "type": 132,
            "name": "alpha",
            "columns": 1,
            "times": [0.0],
            "values": [[alpha]],
            "binary_unknown0": 0xFFFF,
            "binary_column_count": 1,
            "binary_unknown1": [0, 0, 0],
        }
        controllers.append(controller)
        node.controllers = controllers
        return
    controller["type"] = 132
    controller["name"] = "alpha"
    controller["columns"] = 1
    controller["times"] = [0.0]
    controller["values"] = [[alpha]]
    controller["binary_unknown0"] = int(
        controller.get("binary_unknown0", 0xFFFF)
    )
    controller["binary_column_count"] = 1
    controller["binary_unknown1"] = list(
        controller.get("binary_unknown1", [0, 0, 0])
    )


def align_head_art_anatomy_to_donor(
    *,
    art_part: HeadArtPart,
    anatomy: HeadArtAnatomyReport,
    donor_model: Any,
) -> HeadAlignmentResult:
    """Fit eye/mouth anatomy to the donor's exact native component nodes."""

    nodes = _model_nodes(donor_model)
    anchors: list[HeadAlignmentAnchor] = []
    for role in (
        HeadArtAnatomyRole.LEFT_EYE,
        HeadArtAnatomyRole.RIGHT_EYE,
        HeadArtAnatomyRole.UPPER_TEETH,
        HeadArtAnatomyRole.LOWER_TEETH,
        HeadArtAnatomyRole.TONGUE,
    ):
        component = anatomy.component(role)
        target = _unique_node(nodes, ROLE_NODE_NAMES[role])
        target_world = _node_vertices_in_world(target)
        if not target_world:
            raise HeadFacialTransplantError(
                f"Donor node {target.name!r} contains no alignment geometry"
            )
        anchors.append(
            HeadAlignmentAnchor(
                role.value,
                component.centroid,
                tuple(
                    float(value)
                    for value in np.asarray(target_world, dtype=np.float64).mean(axis=0)
                ),
                weight=2.0 if "eye" in role.value else 1.0,
            )
        )
    return solve_headhook_alignment(
        HeadAlignmentRequest(
            anchors=tuple(anchors),
            headhook_to_body=(
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            scale_mode="similarity",
        )
    )


def _skin_transfer_and_payload_channels(
    *,
    donor_skin: Any,
    payload_skin: Any,
    target_world_vertices: Sequence[Sequence[float]],
    target_world_normals: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    """Use one headhook/world space for transfer and node-local output."""

    donor_world = _node_vertices_in_world(donor_skin)
    payload_vertices, payload_normals = _world_to_node_channels(
        payload_skin,
        target_world_vertices,
        target_world_normals,
    )
    return donor_world, payload_vertices, payload_normals


def _apply_skin_payload(
    node: Any,
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    normals: Sequence[Sequence[float]],
    uvs: Sequence[Sequence[float]],
    rows: Sequence[Any],
    texture_resref: str,
) -> None:
    from src.core.geometry.model_data import BoneWeight, VertexSkinData

    _apply_mesh_channels(
        node,
        vertices=vertices,
        faces=faces,
        normals=normals,
        uvs=uvs,
        texture_resref=texture_resref,
    )
    node.skin_data = [
        VertexSkinData(
            [
                BoneWeight(
                    bone_index=int(influence.palette_slot),
                    weight=float(influence.weight),
                )
                for influence in row.influences
            ]
        )
        for row in rows
    ]
    node.bone_indices = [
        [int(influence.palette_slot) for influence in row.influences]
        for row in rows
    ]
    node.bone_weights = [
        [float(influence.weight) for influence in row.influences]
        for row in rows
    ]


def _apply_rigid_payload(
    node: Any,
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    normals: Sequence[Sequence[float]],
    uvs: Sequence[Sequence[float]],
    texture_resref: str,
    skin_bone_name: str = "",
) -> None:
    _apply_mesh_channels(
        node,
        vertices=vertices,
        faces=faces,
        normals=normals,
        uvs=uvs,
        texture_resref=texture_resref,
    )
    rigid_bone = str(skin_bone_name or "").strip()
    if rigid_bone:
        from src.core.geometry.model_data import BoneWeight, VertexSkinData

        matches = [
            index
            for index, value in enumerate(
                list(getattr(node, "bone_map", ()) or ())
            )
            if str(value).casefold() == rigid_bone.casefold()
        ]
        if len(matches) != 1:
            raise HeadFacialTransplantError(
                f"Secondary skin {node.name!r} must contain exactly one "
                f"{rigid_bone!r} palette slot"
            )
        slot = matches[0]
        node.skin_data = [
            VertexSkinData([BoneWeight(slot, 1.0)])
            for _ in node.vertices
        ]
        node.bone_indices = [[slot] for _ in node.vertices]
        node.bone_weights = [[1.0] for _ in node.vertices]
    else:
        node.skin_data = []
        node.bone_map = []
        node.bone_map_floats = []
        node.bone_indices = []
        node.bone_weights = []
        node.qbone_list = []
        node.tbone_list = []


def _apply_mesh_channels(
    node: Any,
    *,
    vertices: Sequence[Sequence[float]],
    faces: Sequence[Sequence[int]],
    normals: Sequence[Sequence[float]],
    uvs: Sequence[Sequence[float]],
    texture_resref: str,
) -> None:
    node.vertices = [tuple(float(value) for value in row[:3]) for row in vertices]
    node.faces = [tuple(int(value) for value in row[:3]) for row in faces]
    node.normals = [tuple(float(value) for value in row[:3]) for row in normals]
    node.uvs = [tuple(float(value) for value in row[:2]) for row in uvs]
    node.face_uvs = []
    node.face_mats = [0] * len(node.faces)
    node.tangents = []
    node.uvs_lm = []
    node.uvs_2 = []
    node.uvs_3 = []
    node.texture = texture_resref
    node.texture_names = [texture_resref]
    node.tex_count = 1
    node.render = True
    node.alpha = 1.0
    node.ambient = (1.0, 1.0, 1.0)
    node.diffuse = (1.0, 1.0, 1.0)
    node.selfillum = (0.0, 0.0, 0.0)
    node.has_shadow = False
    node.mdx_tangent_space = False
    node.tangent_space = False
    # OBJ UVs and the supplied source image are already in the same authored
    # orientation.  Preview must not apply an additional V inversion.
    node.uv_v_flip = False
    # The generic binary writer interprets ``uv_v_flip=False`` as a request to
    # invert renderer-imported rows during MDX serialization.  Head Builder
    # art is already authored in the final KOTOR orientation, so make the
    # source-preserving write contract explicit.
    setattr(node, "_gr_mdx_uv_transform", "identity")


def _world_to_node_channels(
    node: Any,
    world_vertices: Sequence[Sequence[float]],
    world_normals: Sequence[Sequence[float]],
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    identity = SimpleNamespace(
        world_transform=lambda: (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    rebase = build_head_component_rebase(
        identity,
        node,
        skin_translation_only=False,
    )
    vertices, normals, _tangents = rebase_head_component_channels(
        vertices=world_vertices,
        normals=world_normals,
        tangents=(),
        rebase=rebase,
    )
    return vertices, normals


def _node_vertices_in_world(
    node: Any,
) -> tuple[tuple[float, float, float], ...]:
    identity = SimpleNamespace(
        world_transform=lambda: (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    rebase = build_head_component_rebase(
        node,
        identity,
        skin_translation_only=False,
    )
    vertices, _normals, _tangents = rebase_head_component_channels(
        vertices=tuple(node.vertices),
        normals=(),
        tangents=(),
        rebase=rebase,
    )
    return vertices


def _unique_node(
    nodes: Sequence[Any],
    names: Sequence[str],
    *,
    require_skin: bool = False,
) -> Any:
    wanted = {str(value).casefold() for value in names}
    matches = [
        node
        for node in nodes
        if str(getattr(node, "name", "")).casefold() in wanted
        and (
            not require_skin
            or bool(getattr(node, "skin_data", ()) or ())
        )
    ]
    if len(matches) != 1:
        raise HeadFacialTransplantError(
            f"Expected one donor node from {tuple(names)!r}; found {len(matches)}"
        )
    return matches[0]


def _palette_slot(palette: Sequence[str], name: str) -> int:
    matches = [
        index
        for index, value in enumerate(palette)
        if str(value).casefold() == str(name).casefold()
    ]
    if len(matches) != 1:
        raise HeadFacialTransplantError(
            f"Donor palette must contain exactly one {name!r} slot"
        )
    return matches[0]


def _model_nodes(model: Any) -> list[Any]:
    all_nodes = getattr(model, "all_nodes", None)
    return list(all_nodes()) if callable(all_nodes) else []


__all__ = [
    "HeadFacialTransplantError",
    "HeadFacialTransplantReport",
    "HeadFacialTransplantResult",
    "ROLE_NODE_NAMES",
    "align_head_art_anatomy_to_donor",
    "build_head_facial_transplant",
]
