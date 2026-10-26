"""Workflow facade connecting Placeable Builder assets to Map Studio resources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.core.placeables.placeable_utp_io import (
    PlaceableBundleIssue,
    PlaceableBundleResource,
    PlaceableResourceCollisionError,
    build_placeable_resource_bundle,
    export_placeable_utp,
    interactive_resource_dependency_keys,
    read_placeable_utp,
    validate_placeable_resource_bundle,
)
from src.core.project.placeable_asset import PlaceableAsset, load_placeable_asset
from src.core.project.resource_address import ResourceAddress
from src.core.secondary_motion.profile_io import serialize_secondary_motion_profile
from src.core.secondary_motion.workflow import compile_secondary_motion_profile
from src.core.resources.game_resource_provider import GameResourceProvider, GameResourceQuery
from src.core.resources.placeable_library import discover_placeable_library_rows
from src.core.workflow.particle_placeable_export import build_particle_placeable


@dataclass(frozen=True)
class PlaceableReferencedResourcesResult:
    resources: tuple[tuple[str, str, bytes], ...]
    issues: tuple[PlaceableBundleIssue, ...]
    selected_template_resrefs: tuple[str, ...]
    missing_template_resrefs: tuple[str, ...]
    secondary_motion_profiles: tuple[tuple[str, bytes], ...] = ()
    engine_ready: bool = False

    @property
    def has_blocking(self) -> bool:
        return any(issue.severity == "blocking" for issue in self.issues)


def placeable_library_rows(
    root: str | Path,
    *,
    game: str = "",
    provider: GameResourceProvider | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return Map Studio-compatible stock and project UTP rows."""

    roots = (root,) if str(root or "").strip() else ()
    return discover_placeable_library_rows(asset_roots=roots, provider=provider, game=game)


def _provider_reader(provider: GameResourceProvider | None) -> Callable[[Any], bytes] | None:
    return provider.read_bytes if provider is not None else None


def _appearance_bytes(provider: GameResourceProvider | None, game: str) -> bytes | None:
    if provider is None:
        return None
    try:
        return provider.read_bytes(GameResourceQuery(game=game or None, resref="placeables", restype="2DA"))
    except Exception:
        return None


def referenced_placeable_resource_report(
    root: str | Path,
    template_resrefs: Iterable[str],
    *,
    game: str = "",
    provider: GameResourceProvider | None = None,
    resource_reader: Callable[[Any], bytes] | None = None,
    base_resolver: Callable[[PlaceableAsset], bytes | None] | None = None,
    appearance_2da_bytes: bytes | None = None,
) -> PlaceableReferencedResourcesResult:
    """Resolve selected project/stock placeables into collision-safe resources."""

    wanted = tuple(dict.fromkeys(str(value or "").strip().lower() for value in template_resrefs if str(value or "").strip()))
    rows = placeable_library_rows(root, game=game, provider=provider)
    by_resref = {str(row.get("resref") or "").lower(): row for row in rows}
    reader = resource_reader or _provider_reader(provider)
    appearance_bytes = appearance_2da_bytes or _appearance_bytes(provider, str(game or "").upper())
    collected: dict[tuple[str, str], PlaceableBundleResource] = {}
    issues: list[PlaceableBundleIssue] = []
    selected: list[str] = []
    missing: list[str] = []
    externally_resolved_dependencies: set[tuple[str, str]] = set()
    particle_appearance_changed = False
    secondary_motion_profiles: dict[str, bytes] = {}

    def add(resource: PlaceableBundleResource) -> None:
        prior = collected.get(resource.key)
        if prior is not None and bytes(prior.data) != bytes(resource.data):
            raise PlaceableResourceCollisionError(
                f"Resource collision for {resource.key[0]}.{resource.key[1].lower()} while resolving Placeable Builder assets."
            )
        collected.setdefault(resource.key, resource)

    def add_secondary_motion(asset: PlaceableAsset) -> None:
        compiled = compile_secondary_motion_profile(asset.secondary_motion, target_kind="placeable")
        profile_id = str(compiled.get("profile_id") or "secondary_motion")
        payload = serialize_secondary_motion_profile(compiled)
        prior = secondary_motion_profiles.get(profile_id)
        if prior is not None and prior != payload:
            raise PlaceableResourceCollisionError(
                f"Secondary-motion profile collision for {profile_id!r} while resolving Placeable Builder assets."
            )
        secondary_motion_profiles.setdefault(profile_id, payload)

    for resref in wanted:
        row = by_resref.get(resref)
        if row is None:
            missing.append(resref)
            issues.append(PlaceableBundleIssue("blocking", "missing_placeable_asset", f"No placeable library row resolves '{resref}'.", (resref, "UTP")))
            continue
        path = str(row.get("path") or "")
        if row.get("source") == "placeable_builder" and path:
            try:
                asset = load_placeable_asset(path)
                add_secondary_motion(asset)
                base_bytes = base_resolver(asset) if base_resolver is not None else None
                if base_bytes is None and provider is not None and asset.base_template is not None:
                    base_bytes = provider.read_bytes(asset.base_template)
                particle_effects = tuple((asset.metadata or {}).get("particle_effects") or ())
                if particle_effects:
                    if reader is None:
                        raise ValueError(
                            f"{resref} has particle effects, but no target-game resource reader is connected."
                        )
                    build = build_particle_placeable(
                        asset,
                        base_utp_bytes=base_bytes,
                        appearance_2da_bytes=bytes(appearance_bytes or b""),
                        resource_reader=reader,
                    )
                    appearance_bytes = build.appearance_2da_bytes
                    particle_appearance_changed = True
                    exported = build.utp_result
                    for resource in build.resources:
                        add(resource)
                    issues.extend(build.issues)
                else:
                    exported = export_placeable_utp(
                        asset,
                        base_utp_bytes=base_bytes,
                        appearance_2da_bytes=appearance_bytes,
                    )
                    if reader is not None:
                        bundle = build_placeable_resource_bundle(asset, exported, resource_reader=reader)
                        for resource in bundle.resources:
                            add(resource)
                    else:
                        add(PlaceableBundleResource(exported.template_resref, "UTP", exported.utp_bytes, path))
                        if any((asset.resources.mdl, asset.resources.mdx, asset.resources.pwk, *asset.resources.textures)):
                            issues.append(
                                PlaceableBundleIssue(
                                    "blocking",
                                    "placeable_resource_reader_missing",
                                    f"{resref} references model/texture resources but no resource reader was provided.",
                                    (resref, "UTP"),
                                )
                            )
                issues.extend(PlaceableBundleIssue("warning", "placeable_export_unproven", warning, (resref, "UTP")) for warning in exported.warnings)
                if reader is not None:
                    for dependency_resref, dependency_type in exported.readback.dependency_keys:
                        dependency_key = (dependency_resref.lower(), dependency_type.upper())
                        if dependency_key in collected:
                            continue
                        try:
                            dependency_data = reader(
                                GameResourceQuery(
                                    game=str(game or asset.game or "").upper() or None,
                                    resref=dependency_resref,
                                    restype=dependency_type,
                                )
                            )
                            if dependency_data:
                                externally_resolved_dependencies.add(dependency_key)
                        except Exception:
                            pass
                selected.append(resref)
            except Exception as exc:
                issues.append(PlaceableBundleIssue("blocking", "placeable_export_failed", str(exc), (resref, "UTP")))
            continue

        if provider is None:
            missing.append(resref)
            issues.append(PlaceableBundleIssue("blocking", "stock_utp_provider_missing", f"Stock placeable '{resref}' needs a game resource provider.", (resref, "UTP")))
            continue
        try:
            data = provider.read_bytes(GameResourceQuery(game=game or row.get("game") or None, resref=resref, restype="UTP"))
            read_placeable_utp(data)
            add(PlaceableBundleResource(resref, "UTP", data, str(row.get("source") or "game_resource")))
            selected.append(resref)
        except Exception as exc:
            issues.append(PlaceableBundleIssue("blocking", "stock_utp_read_failed", str(exc), (resref, "UTP")))

    if particle_appearance_changed and appearance_bytes:
        add(
            PlaceableBundleResource(
                "placeables",
                "2DA",
                bytes(appearance_bytes),
                "generated_particle_placeable_appearance_table",
            )
        )

    for issue in validate_placeable_resource_bundle(collected.values()):
        key = tuple(issue.resource_key or ())
        if issue.code == "missing_placeable_dependency" and key in externally_resolved_dependencies:
            issues.append(
                PlaceableBundleIssue(
                    "warning",
                    "placeable_dependency_external_resolved",
                    f"{key[0]}.{key[1].lower()} resolves from the selected game/Override but is not bundled in the module.",
                    key,
                )
            )
        else:
            issues.append(issue)
    resources = tuple(resource.as_tuple() for _key, resource in sorted(collected.items()))
    return PlaceableReferencedResourcesResult(
        resources=resources,
        issues=tuple(issues),
        selected_template_resrefs=tuple(selected),
        missing_template_resrefs=tuple(missing),
        secondary_motion_profiles=tuple(sorted(secondary_motion_profiles.items())),
        engine_ready=False,
    )


def referenced_placeable_resources(
    root: str | Path,
    template_resrefs: Iterable[str],
    **kwargs: Any,
) -> tuple[tuple[str, str, bytes], ...]:
    """Convenience API for authored-module ``extra_resources`` injection."""

    return referenced_placeable_resource_report(root, template_resrefs, **kwargs).resources


def referenced_interactive_resource_report(
    root: str | Path,
    template_requests: Iterable[tuple[str, str] | str],
    *,
    game: str = "",
    provider: GameResourceProvider | None = None,
) -> PlaceableReferencedResourcesResult:
    """Resolve manually placed UTP/UTD templates and their declared graph.

    Base-game templates remain external. Module, Override, project, and unknown
    provenance is bundled into the authored MOD so moving a template into a new
    module (notably ``plcaa``) cannot silently lose its local UTP/UTD, DLG, NCS,
    or UTI dependencies.
    """

    normalized: list[tuple[str, str]] = []
    for value in tuple(template_requests or ()):
        if isinstance(value, str):
            resref, restype = value, "UTP"
        else:
            try:
                resref, restype = value[:2]
            except (TypeError, ValueError):
                continue
        key = (str(resref or "").strip().lower(), str(restype or "UTP").strip().upper().lstrip("."))
        if key[0] and key[1] in {"UTP", "UTD"} and key not in normalized:
            normalized.append(key)

    rows = placeable_library_rows(root, game=game, provider=provider)
    by_key = {
        (
            str(row.get("resref") or row.get("template_resref") or "").strip().lower(),
            str(row.get("restype") or row.get("resource_type") or "UTP").strip().upper().lstrip("."),
        ): row
        for row in rows
    }
    resources: dict[tuple[str, str], tuple[str, str, bytes]] = {}
    issues: list[PlaceableBundleIssue] = []
    selected: list[str] = []
    missing: list[str] = []
    scripted_graph_partial = False
    secondary_motion_profiles: dict[str, bytes] = {}

    def add_resource(resref: str, restype: str, data: bytes) -> None:
        key = (str(resref).lower(), str(restype).upper().lstrip("."))
        row = (key[0], f".{key[1]}", bytes(data))
        prior = resources.get(key)
        if prior is not None and prior[2] != row[2]:
            raise PlaceableResourceCollisionError(
                f"Resource collision for {key[0]}.{key[1].lower()} while resolving interactive templates."
            )
        resources.setdefault(key, row)

    authored_utps = [
        resref
        for resref, restype in normalized
        if restype == "UTP" and str((by_key.get((resref, restype)) or {}).get("source") or "") == "placeable_builder"
    ]
    if authored_utps:
        authored = referenced_placeable_resource_report(root, authored_utps, game=game, provider=provider)
        issues.extend(authored.issues)
        for resref, restype, data in authored.resources:
            add_resource(resref, restype, data)
        selected.extend(authored.selected_template_resrefs)
        missing.extend(authored.missing_template_resrefs)
        for profile_id, payload in authored.secondary_motion_profiles:
            prior = secondary_motion_profiles.get(profile_id)
            if prior is not None and prior != payload:
                raise PlaceableResourceCollisionError(
                    f"Secondary-motion profile collision for {profile_id!r} while resolving interactive templates."
                )
            secondary_motion_profiles.setdefault(profile_id, payload)

    def row_address(row: dict[str, Any]) -> ResourceAddress | None:
        metadata = dict(row.get("metadata") or {})
        value = metadata.get("address")
        try:
            return ResourceAddress.from_dict(value) if value else None
        except Exception:
            return None

    def resolve(resref: str, restype: str, preferred: ResourceAddress | None = None):
        if provider is None:
            raise FileNotFoundError("A target-game resource provider is required.")
        attempts: list[Any] = []
        if preferred is not None:
            attempts.append(
                ResourceAddress(
                    scheme="module_resource" if preferred.module_id else preferred.scheme,
                    game=preferred.game or game,
                    module_id=preferred.module_id,
                    resref=resref,
                    restype=restype,
                    layer=preferred.layer if preferred.module_id else None,
                    path=preferred.path if preferred.module_id else None,
                )
            )
        attempts.append(GameResourceQuery(game=game or None, resref=resref, restype=restype))
        last_error: Exception | None = None
        for query in attempts:
            try:
                return provider.resolve(query)
            except Exception as exc:
                last_error = exc
        raise last_error or FileNotFoundError(f"Missing {resref}.{restype.lower()}.")

    queue: list[tuple[str, str, ResourceAddress | None, str]] = []
    visited: set[tuple[str, str, str, str]] = set()
    for resref, restype in normalized:
        row = by_key.get((resref, restype))
        if row is None:
            missing.append(resref)
            issues.append(
                PlaceableBundleIssue(
                    "blocking",
                    "missing_interactive_template",
                    f"No target-game {restype} template row resolves '{resref}'.",
                    (resref, restype),
                )
            )
            continue
        if str(row.get("source") or "") == "placeable_builder":
            continue
        try:
            result = resolve(resref, restype, row_address(row))
            dependencies = interactive_resource_dependency_keys(restype, result.data)
        except Exception as exc:
            issues.append(PlaceableBundleIssue("blocking", "interactive_template_read_failed", str(exc), (resref, restype)))
            continue
        selected.append(resref)
        layer = str(result.address.layer or "unknown").lower()
        if layer not in {"base", "texturepack"}:
            add_resource(resref, restype, result.data)
        for dependency_resref, dependency_type in dependencies:
            queue.append((dependency_resref, dependency_type, result.address, resref))

    while queue:
        resref, restype, preferred, owner = queue.pop(0)
        identity = (
            resref.lower(),
            restype.upper(),
            str(getattr(preferred, "module_id", "") or ""),
            str(getattr(preferred, "layer", "") or ""),
        )
        if identity in visited:
            continue
        visited.add(identity)
        try:
            result = resolve(resref, restype, preferred)
        except Exception:
            issues.append(
                PlaceableBundleIssue(
                    "blocking",
                    "missing_interactive_dependency",
                    f"{owner} requires missing {resref}.{restype.lower()} in the target game/module resource graph.",
                    (resref, restype),
                )
            )
            continue
        layer = str(result.address.layer or "unknown").lower()
        if layer not in {"base", "texturepack"}:
            add_resource(resref, restype, result.data)
        if restype.upper() == "NCS":
            scripted_graph_partial = True
        try:
            nested = interactive_resource_dependency_keys(restype, result.data)
        except Exception as exc:
            issues.append(
                PlaceableBundleIssue(
                    "blocking",
                    "invalid_interactive_dependency",
                    f"Could not read {resref}.{restype.lower()} required by {owner}: {exc}",
                    (resref, restype),
                )
            )
            continue
        for nested_resref, nested_type in nested:
            queue.append((nested_resref, nested_type, result.address, resref))

    if scripted_graph_partial:
        issues.append(
            PlaceableBundleIssue(
                "warning",
                "compiled_script_graph_requires_game_proof",
                "Declared NCS resources resolve, but compiled scripts can still depend on tags, globals, triggers, and other module objects; composite puzzles remain unproven until the plcaa interaction test passes.",
            )
        )
    return PlaceableReferencedResourcesResult(
        resources=tuple(resources[key] for key in sorted(resources)),
        issues=tuple(issues),
        selected_template_resrefs=tuple(dict.fromkeys(selected)),
        missing_template_resrefs=tuple(dict.fromkeys(missing)),
        secondary_motion_profiles=tuple(sorted(secondary_motion_profiles.items())),
        engine_ready=False,
    )


__all__ = [
    "PlaceableReferencedResourcesResult",
    "placeable_library_rows",
    "referenced_placeable_resource_report",
    "referenced_placeable_resources",
    "referenced_interactive_resource_report",
]
