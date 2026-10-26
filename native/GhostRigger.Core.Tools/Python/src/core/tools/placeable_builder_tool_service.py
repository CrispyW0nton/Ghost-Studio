"""Non-Qt orchestration for the Placeable Builder product workbench."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

from pykotor.resource.formats.gff import GFFContent, read_gff
from pykotor.resource.formats.twoda import read_2da
from pykotor.resource.generics.utp import read_utp

from src.core.placeables.placeable_utp_io import (
    PlaceableBundleResource,
    PlaceableUTPExportResult,
    build_placeable_resource_bundle,
    export_placeable_utp,
)
from src.core.project.placeable_asset import (
    PLACEABLE_ASSET_FILE_SUFFIX,
    PLACEABLE_SCRIPT_HOOKS,
    PlaceableAppearanceMappingEvidence,
    PlaceableAsset,
    PlaceableAssetValidation,
    PlaceableBaseTemplateEvidence,
    PlaceableGameplay,
    load_placeable_asset,
    save_placeable_asset,
    validate_placeable_asset,
)
from src.core.project.project_validation import ProjectValidationIssue
from src.core.project.resource_address import ResourceAddress
from src.core.secondary_motion.profile_io import serialize_secondary_motion_profile
from src.core.secondary_motion.workflow import compile_secondary_motion_profile
from src.core.validation.secondary_motion_validator import validate_secondary_motion_profile
from src.core.resources.game_resource_provider import GameResourceProvider, GameResourceQuery
from src.core.workflow.particle_placeable_export import build_particle_placeable
from src.core.workflow.placeable_builder_service import placeable_library_rows


@dataclass(frozen=True)
class PlaceableBuilderSaveResult:
    ok: bool
    asset: PlaceableAsset
    validation: PlaceableAssetValidation
    sidecar_path: str = ""
    utp_path: str = ""
    utp_result: PlaceableUTPExportResult | None = None
    bundle_dir: str = ""
    bundle_resources: tuple[tuple[str, str, int], ...] = ()
    messages: tuple[str, ...] = ()
    engine_ready: bool = False


def _asset(value: PlaceableAsset | Mapping[str, Any]) -> PlaceableAsset:
    if isinstance(value, PlaceableAsset):
        return PlaceableAsset.from_dict(value.to_dict())
    return PlaceableAsset.from_dict(dict(value or {}))


def _safe_copy_resref(value: str) -> str:
    clean = "".join(char for char in str(value or "").strip().lower() if char.isalnum() or char == "_")
    clean = clean or "placeable"
    suffix = "_copy"
    return f"{clean[: 16 - len(suffix)]}{suffix}"


class PlaceableBuilderToolService:
    """Coordinate documents, providers, UTP output, and evidence without Qt."""

    def __init__(self, library_root: str | Path, *, provider: GameResourceProvider | None = None) -> None:
        self.library_root = Path(library_root)
        self.provider = provider

    def set_library_root(self, root: str | Path) -> None:
        self.library_root = Path(root)

    def set_provider(self, provider: GameResourceProvider | None) -> None:
        self.provider = provider

    def rows(self, *, game: str = "") -> tuple[dict[str, Any], ...]:
        return placeable_library_rows(self.library_root, game=game, provider=self.provider)

    @staticmethod
    def new_asset(*, game: str = "K2") -> PlaceableAsset:
        return PlaceableAsset(game=str(game or "K2").upper())

    @staticmethod
    def clone_asset(value: PlaceableAsset | Mapping[str, Any]) -> PlaceableAsset:
        source = _asset(value)
        data = source.to_dict()
        fresh = PlaceableAsset()
        data["asset_id"] = fresh.asset_id
        data["template_resref"] = _safe_copy_resref(source.template_resref)
        data["tag"] = data["template_resref"]
        data["display_name"] = f"{source.display_name or source.template_resref or 'Placeable'} Copy"
        metadata = dict(data.get("metadata") or {})
        metadata["cloned_from_asset_id"] = source.asset_id
        metadata["cloned_from_template_resref"] = source.template_resref
        data["metadata"] = metadata
        return PlaceableAsset.from_dict(data)

    def _read(self, query: GameResourceQuery | ResourceAddress) -> bytes | None:
        if self.provider is None:
            return None
        try:
            data = self.provider.read_bytes(query)
        except Exception:
            return None
        return bytes(data) if data else None

    def _placeables_2da(self, game: str) -> bytes | None:
        return self._read(GameResourceQuery(game=game, resref="placeables", restype="2DA"))

    def _appearance_model(self, game: str, appearance_id: int | None) -> tuple[str, bytes | None]:
        data = self._placeables_2da(game)
        if data is None or appearance_id is None or appearance_id < 0:
            return "", data
        try:
            table = read_2da(data)
            model = str(table.get_cell(int(appearance_id), "modelname") or "").strip().lower()
        except Exception:
            model = ""
        return ("" if model in {"", "****", "null"} else model), data

    def _base_bytes(self, asset: PlaceableAsset) -> bytes | None:
        if asset.base_template is None:
            return None
        return self._read(asset.base_template)

    def _with_provider_evidence(self, value: PlaceableAsset | Mapping[str, Any]) -> PlaceableAsset:
        asset = _asset(value)
        base_bytes = self._base_bytes(asset)
        if base_bytes:
            try:
                gff = read_gff(base_bytes)
                if gff.content == GFFContent.UTP:
                    asset.base_evidence = PlaceableBaseTemplateEvidence(
                        template=asset.base_template,
                        sha256=sha256(base_bytes).hexdigest(),
                        field_count=len(gff.root),
                        source=asset.base_template.stable_key() if asset.base_template else "",
                    )
            except Exception:
                pass
        model_resref, appearance_bytes = self._appearance_model(asset.game, asset.appearance_id)
        expected_model = model_resref
        if asset.resources.mdl is not None and asset.resources.mdl.resref:
            expected_model = str(asset.resources.mdl.resref).lower()
        if appearance_bytes and model_resref and expected_model == model_resref:
            asset.appearance_evidence = PlaceableAppearanceMappingEvidence(
                game=asset.game,
                appearance_id=asset.appearance_id,
                model_resref=model_resref,
                source=f"{asset.game}:placeables.2da",
                source_sha256=sha256(appearance_bytes).hexdigest(),
                verified=True,
            )
        identity = dict(asset.secondary_motion.get("model_identity") or {})
        identity["target_kind"] = "placeable"
        if expected_model:
            identity["model_resrefs"] = [expected_model]
        for field_name, address in (("mdl_sha256", asset.resources.mdl), ("mdx_sha256", asset.resources.mdx)):
            if address is None:
                continue
            payload = self._read(address)
            if payload:
                identity[field_name] = sha256(payload).hexdigest()
        asset.secondary_motion["model_identity"] = identity
        runtime = dict(asset.secondary_motion.get("runtime") or {})
        runtime.update({"state": "capture_only", "write_enabled": False})
        asset.secondary_motion["runtime"] = runtime
        return asset

    @staticmethod
    def _validation(asset: PlaceableAsset) -> PlaceableAssetValidation:
        base = validate_placeable_asset(asset)
        secondary = validate_secondary_motion_profile(asset.secondary_motion, target_kind="placeable")
        issues = list(base.issues)
        for issue in secondary.issues:
            issues.append(ProjectValidationIssue(
                severity="blocking" if issue.severity == "blocking" else "warning",
                code=f"secondary_motion_{issue.code}",
                message=issue.message,
                target=f"secondary_motion.{issue.target}" if issue.target else "secondary_motion",
            ))
        document_valid = base.document_valid and secondary.can_export_profile
        return PlaceableAssetValidation(
            issues=tuple(issues),
            document_valid=document_valid,
            utp_export_ready=base.utp_export_ready and secondary.can_export_profile,
            structural_evidence_ready=base.structural_evidence_ready and secondary.can_export_profile,
            engine_ready=False,
        )

    def validate(self, value: PlaceableAsset | Mapping[str, Any]) -> PlaceableAssetValidation:
        return self._validation(self._with_provider_evidence(value))

    def load_row(self, row: Mapping[str, Any]) -> PlaceableAsset:
        data = dict(row or {})
        path = str(data.get("path") or "")
        if str(data.get("source") or "") == "placeable_builder" and path:
            return load_placeable_asset(path)
        if self.provider is None:
            raise ValueError("A KOTOR resource provider is required to clone a stock placeable.")
        game = str(data.get("game") or "K2").upper()
        resref = str(data.get("resref") or data.get("template_resref") or "").strip().lower()
        if not resref:
            raise ValueError("The selected library row has no UTP resref.")
        raw = self.provider.read_bytes(GameResourceQuery(game=game, resref=resref, restype="UTP"))
        gff = read_gff(raw)
        if gff.content != GFFContent.UTP:
            raise ValueError(f"{resref}.utp is not a UTP GFF resource.")
        utp = read_utp(raw)
        address_data = dict((data.get("metadata") or {}).get("address") or {})
        base_address = (
            ResourceAddress.from_dict(address_data)
            if address_data
            else ResourceAddress(
                scheme="game_resource",
                game=game,
                layer="base",
                resref=resref,
                restype="UTP",
            )
        )
        gameplay = PlaceableGameplay(
            static=bool(utp.static),
            useable=bool(utp.useable),
            has_inventory=bool(utp.has_inventory),
            inventory_items=[str(item.resref).lower() for item in utp.inventory],
            lockable=bool(utp.lockable),
            locked=bool(utp.locked),
            key_required=bool(utp.key_required),
            key_name=str(utp.key_name or "").lower(),
            auto_remove_key=bool(utp.auto_remove_key),
            unlock_dc=int(utp.unlock_dc),
            lock_dc=int(utp.lock_dc),
            trap_detectable=bool(utp.trap_detectable),
            trap_detect_dc=int(utp.trap_detect_dc),
            trap_disarmable=bool(utp.trap_disarmable),
            trap_disarm_dc=int(utp.trap_disarm_dc),
            trap_flag=int(utp.trap_flag),
            trap_one_shot=bool(utp.trap_one_shot),
            trap_type=int(utp.trap_type),
            maximum_hp=int(utp.maximum_hp),
            current_hp=int(utp.current_hp),
            hardness=int(utp.hardness),
            plot=bool(utp.plot),
            min1_hp=bool(utp.min1_hp),
            not_blastable=bool(utp.not_blastable),
            party_interact=bool(utp.party_interact),
            conversation_resref=str(utp.conversation).lower(),
        )
        scripts = {
            hook: str(getattr(utp, hook)).lower()
            for hook in PLACEABLE_SCRIPT_HOOKS
            if str(getattr(utp, hook))
        }
        asset = PlaceableAsset(
            game=game,
            template_resref=resref,
            tag=str(utp.tag or resref),
            display_name=str(utp.name or resref),
            description=str(utp.description or ""),
            comment=str(utp.comment or ""),
            category=str(data.get("subcategory") or "decor").strip().lower() or "decor",
            visual_source="stock",
            appearance_id=int(utp.appearance_id),
            gameplay=gameplay,
            scripts=scripts,
            base_template=base_address,
            base_evidence=PlaceableBaseTemplateEvidence(
                template=base_address,
                sha256=sha256(raw).hexdigest(),
                field_count=len(gff.root),
                source=str(data.get("source") or base_address.stable_key()),
            ),
            metadata={"cloned_from_stock_utp": resref, "source_library_row": data},
        )
        return self._with_provider_evidence(asset)

    def save(self, value: PlaceableAsset | Mapping[str, Any]) -> PlaceableBuilderSaveResult:
        asset = self._with_provider_evidence(value)
        validation = self._validation(asset)
        if not validation.utp_export_ready:
            messages = tuple(issue.message for issue in validation.issues if issue.severity == "blocking")
            return PlaceableBuilderSaveResult(False, asset, validation, messages=messages)
        self.library_root.mkdir(parents=True, exist_ok=True)
        sidecar = self.library_root / f"{asset.template_resref}{PLACEABLE_ASSET_FILE_SUFFIX}"
        utp_path = self.library_root / f"{asset.template_resref}.utp"
        try:
            utp_result = export_placeable_utp(
                asset,
                base_utp_bytes=self._base_bytes(asset),
                appearance_2da_bytes=self._placeables_2da(asset.game),
                output_path=utp_path,
            )
            save_placeable_asset(asset, sidecar)
        except Exception as exc:
            return PlaceableBuilderSaveResult(False, asset, validation, messages=(str(exc),))
        messages = tuple(utp_result.warnings)
        return PlaceableBuilderSaveResult(
            True,
            asset,
            self._validation(asset),
            sidecar_path=str(sidecar),
            utp_path=str(utp_path),
            utp_result=utp_result,
            messages=messages,
            engine_ready=False,
        )

    def export_utp(
        self,
        value: PlaceableAsset | Mapping[str, Any],
        output_path: str | Path,
    ) -> PlaceableBuilderSaveResult:
        asset = self._with_provider_evidence(value)
        validation = self._validation(asset)
        if not validation.utp_export_ready:
            messages = tuple(issue.message for issue in validation.issues if issue.severity == "blocking")
            return PlaceableBuilderSaveResult(False, asset, validation, messages=messages)
        try:
            result = export_placeable_utp(
                asset,
                base_utp_bytes=self._base_bytes(asset),
                appearance_2da_bytes=self._placeables_2da(asset.game),
                output_path=output_path,
            )
        except Exception as exc:
            return PlaceableBuilderSaveResult(False, asset, validation, messages=(str(exc),))
        return PlaceableBuilderSaveResult(
            True,
            asset,
            self._validation(asset),
            utp_path=str(output_path),
            utp_result=result,
            messages=tuple(result.warnings),
            engine_ready=False,
        )

    def export_game_bundle(
        self,
        value: PlaceableAsset | Mapping[str, Any],
        output_parent: str | Path,
    ) -> PlaceableBuilderSaveResult:
        """Write an install-shaped Override bundle without touching the game."""

        asset = self._with_provider_evidence(value)
        validation = self._validation(asset)
        if not validation.utp_export_ready:
            messages = tuple(issue.message for issue in validation.issues if issue.severity == "blocking")
            return PlaceableBuilderSaveResult(False, asset, validation, messages=messages)
        parent = Path(output_parent)
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / f"{asset.template_resref}_game_bundle"
        if target.exists():
            for index in range(2, 1000):
                candidate = parent / f"{asset.template_resref}_game_bundle_{index}"
                if not candidate.exists():
                    target = candidate
                    break
            else:
                return PlaceableBuilderSaveResult(
                    False,
                    asset,
                    validation,
                    messages=("Could not allocate a new game-bundle output folder.",),
                )
        if self.provider is None:
            return PlaceableBuilderSaveResult(
                False,
                asset,
                validation,
                messages=("Connect the target KOTOR installation before exporting a game bundle.",),
            )

        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(parent)))
        try:
            appearance = self._placeables_2da(asset.game)
            particle_effects = tuple((asset.metadata or {}).get("particle_effects") or ())
            particle_build = None
            resources: list[PlaceableBundleResource]
            if particle_effects:
                particle_build = build_particle_placeable(
                    asset,
                    base_utp_bytes=self._base_bytes(asset),
                    appearance_2da_bytes=bytes(appearance or b""),
                    resource_reader=self.provider.read_bytes,
                )
                utp_result = particle_build.utp_result
                resources = [
                    *particle_build.resources,
                    PlaceableBundleResource(
                        "placeables",
                        "2DA",
                        particle_build.appearance_2da_bytes,
                        "generated_particle_placeable_appearance_table",
                    ),
                ]
                messages = tuple(issue.message for issue in particle_build.issues)
            else:
                utp_result = export_placeable_utp(
                    asset,
                    base_utp_bytes=self._base_bytes(asset),
                    appearance_2da_bytes=appearance,
                )
                bundle = build_placeable_resource_bundle(
                    asset,
                    utp_result,
                    resource_reader=self.provider.read_bytes,
                )
                resources = list(bundle.resources)
                messages = tuple(utp_result.warnings) + tuple(issue.message for issue in bundle.issues)

            unique: dict[tuple[str, str], PlaceableBundleResource] = {}
            for resource in resources:
                prior = unique.get(resource.key)
                if prior is not None and prior.data != resource.data:
                    raise ValueError(f"Bundle collision for {resource.key[0]}.{resource.key[1].lower()}.")
                unique.setdefault(resource.key, resource)
            override_dir = temporary / "install" / "Override"
            override_dir.mkdir(parents=True, exist_ok=True)
            inventory: list[dict[str, Any]] = []
            for key, resource in sorted(unique.items()):
                path = override_dir / f"{key[0]}.{key[1].lower()}"
                path.write_bytes(resource.data)
                if path.read_bytes() != resource.data:
                    raise IOError(f"Bundle write verification failed: {path.name}")
                inventory.append(
                    {
                        "resref": key[0],
                        "restype": key[1],
                        "size": len(resource.data),
                        "sha256": sha256(resource.data).hexdigest(),
                        "source": resource.source,
                    }
                )
            manifest = {
                "schema": "ghostrigger.particle_placeable_game_bundle.v1",
                "game": asset.game,
                "template_resref": asset.template_resref,
                "particle_effect_count": len(particle_effects),
                "appearance_id": particle_build.appearance_id if particle_build is not None else asset.appearance_id,
                "model_resref": particle_build.model_resref if particle_build is not None else "",
                "source_model_resref": particle_build.source_model_resref if particle_build is not None else "",
                "emitter_count": particle_build.emitter_count if particle_build is not None else 0,
                "install_root": "install/Override",
                "resources": inventory,
                "secondary_motion": compile_secondary_motion_profile(
                    asset.secondary_motion,
                    target_kind="placeable",
                ),
                "verification": {
                    "utp_readback": True,
                    "mdl_mdx_readback": bool(particle_effects),
                    "placeables_2da_readback": bool(particle_effects),
                    "in_game_proof": False,
                },
            }
            compiled_secondary_motion = dict(manifest["secondary_motion"])
            profile_id = str(compiled_secondary_motion.get("profile_id") or "secondary_motion")
            safe_profile_id = "".join(
                character if character.isalnum() or character in {"_", "-", "."} else "_"
                for character in profile_id
            ).strip("._") or "secondary_motion"
            profile_path = temporary / "secondary-motion" / f"{safe_profile_id}.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_bytes(serialize_secondary_motion_profile(compiled_secondary_motion))
            manifest["secondary_motion_profile"] = profile_path.relative_to(temporary).as_posix()
            manifest["runtime_dependencies"] = (
                ["cloth-dynamics-k2"]
                if any(str(region.get("backend") or "") == "runtime_pbd" for region in compiled_secondary_motion.get("regions") or ())
                else []
            )
            (temporary / "particle_placeable_bundle.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
            return PlaceableBuilderSaveResult(
                True,
                asset,
                validation,
                utp_path=str(target / "install" / "Override" / f"{asset.template_resref}.utp"),
                utp_result=utp_result,
                bundle_dir=str(target),
                bundle_resources=tuple(
                    (str(row["resref"]), str(row["restype"]), int(row["size"])) for row in inventory
                ),
                messages=messages,
                engine_ready=False,
            )
        except Exception as exc:
            return PlaceableBuilderSaveResult(False, asset, validation, messages=(str(exc),))
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)

    def preview_model_resref(self, value: PlaceableAsset | Mapping[str, Any]) -> str:
        asset = self._with_provider_evidence(value)
        if asset.resources.mdl is not None and asset.resources.mdl.resref:
            return str(asset.resources.mdl.resref).lower()
        if asset.appearance_evidence and asset.appearance_evidence.model_resref:
            return str(asset.appearance_evidence.model_resref).lower()
        model, _data = self._appearance_model(asset.game, asset.appearance_id)
        return model


__all__ = ["PlaceableBuilderSaveResult", "PlaceableBuilderToolService"]
