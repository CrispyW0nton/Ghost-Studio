"""Versioned, human-readable Placeable Builder asset contracts."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .ghostrigger_project import stable_project_id
from .project_validation import ProjectValidationIssue, ProjectValidationReport, validate_resource_address
from .resource_address import ResourceAddress
from .secondary_motion_profile import SecondaryMotionProfile, default_secondary_motion_profile


PLACEABLE_ASSET_FILE_TYPE = "ghostrigger.placeable_asset"
CURRENT_PLACEABLE_ASSET_SCHEMA_VERSION = 2
PLACEABLE_ASSET_FILE_SUFFIX = ".ghostplaceable.json"
PLACEABLE_CATEGORIES = ("container", "terminal", "puzzle", "interactive", "decor")
PLACEABLE_SCRIPT_HOOKS = (
    "on_closed",
    "on_damaged",
    "on_death",
    "on_end_dialog",
    "on_open_failed",
    "on_heartbeat",
    "on_inventory",
    "on_melee_attack",
    "on_force_power",
    "on_open",
    "on_lock",
    "on_unlock",
    "on_used",
    "on_user_defined",
    "on_disarm",
    "on_trap_triggered",
)

_RESREF_RE = re.compile(r"^[A-Za-z0-9_]+$")
_ASSET_ID_RE = re.compile(r"^placeable_[0-9a-f]{32}$")


def _address(value: Any) -> ResourceAddress | None:
    if value in (None, ""):
        return None
    return ResourceAddress.from_dict(value)


def _address_dict(value: ResourceAddress | None) -> dict[str, Any] | None:
    return value.to_dict() if value is not None else None


def _clean_resref(value: Any) -> str:
    return str(value or "").strip().lower()


@dataclass
class PlaceableAppearanceMappingEvidence:
    """Evidence that an Appearance row resolves to the requested model."""

    game: str = ""
    appearance_id: int | None = None
    model_resref: str = ""
    source: str = ""
    source_sha256: str = ""
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "game": str(self.game or "").upper(),
            "appearance_id": self.appearance_id,
            "model_resref": _clean_resref(self.model_resref),
            "source": str(self.source or ""),
            "source_sha256": str(self.source_sha256 or "").lower(),
            "verified": bool(self.verified),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PlaceableAppearanceMappingEvidence | None":
        if not data:
            return None
        value = data.get("appearance_id")
        return cls(
            game=str(data.get("game") or "").upper(),
            appearance_id=int(value) if value is not None else None,
            model_resref=_clean_resref(data.get("model_resref")),
            source=str(data.get("source") or ""),
            source_sha256=str(data.get("source_sha256") or "").lower(),
            verified=bool(data.get("verified")),
        )


@dataclass
class PlaceableBaseTemplateEvidence:
    """Byte identity for a known-loadable UTP used as structural authority."""

    template: ResourceAddress | None = None
    sha256: str = ""
    field_count: int = 0
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "template": _address_dict(self.template),
            "sha256": str(self.sha256 or "").lower(),
            "field_count": int(self.field_count or 0),
            "source": str(self.source or ""),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PlaceableBaseTemplateEvidence | None":
        if not data:
            return None
        return cls(
            template=_address(data.get("template")),
            sha256=str(data.get("sha256") or "").lower(),
            field_count=int(data.get("field_count") or 0),
            source=str(data.get("source") or ""),
        )


@dataclass
class PlaceableGameplay:
    static: bool = False
    useable: bool = True
    has_inventory: bool = False
    inventory_items: list[str] = field(default_factory=list)
    lockable: bool = False
    locked: bool = False
    key_required: bool = False
    key_name: str = ""
    auto_remove_key: bool = False
    unlock_dc: int = 0
    lock_dc: int = 0
    trap_detectable: bool = False
    trap_detect_dc: int = 0
    trap_disarmable: bool = False
    trap_disarm_dc: int = 0
    trap_flag: int = 0
    trap_one_shot: bool = False
    trap_type: int = 0
    maximum_hp: int = 1
    current_hp: int = 1
    hardness: int = 0
    plot: bool = False
    min1_hp: bool = False
    not_blastable: bool = False
    party_interact: bool = False
    conversation_resref: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "static": bool(self.static),
            "useable": bool(self.useable),
            "has_inventory": bool(self.has_inventory),
            "inventory_items": [_clean_resref(value) for value in self.inventory_items],
            "lockable": bool(self.lockable),
            "locked": bool(self.locked),
            "key_required": bool(self.key_required),
            "key_name": _clean_resref(self.key_name),
            "auto_remove_key": bool(self.auto_remove_key),
            "unlock_dc": int(self.unlock_dc),
            "lock_dc": int(self.lock_dc),
            "trap_detectable": bool(self.trap_detectable),
            "trap_detect_dc": int(self.trap_detect_dc),
            "trap_disarmable": bool(self.trap_disarmable),
            "trap_disarm_dc": int(self.trap_disarm_dc),
            "trap_flag": int(self.trap_flag),
            "trap_one_shot": bool(self.trap_one_shot),
            "trap_type": int(self.trap_type),
            "maximum_hp": int(self.maximum_hp),
            "current_hp": int(self.current_hp),
            "hardness": int(self.hardness),
            "plot": bool(self.plot),
            "min1_hp": bool(self.min1_hp),
            "not_blastable": bool(self.not_blastable),
            "party_interact": bool(self.party_interact),
            "conversation_resref": _clean_resref(self.conversation_resref),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PlaceableGameplay":
        data = data or {}
        return cls(
            static=bool(data.get("static")),
            useable=bool(data.get("useable", True)),
            has_inventory=bool(data.get("has_inventory")),
            inventory_items=[_clean_resref(value) for value in data.get("inventory_items") or ()],
            lockable=bool(data.get("lockable")),
            locked=bool(data.get("locked")),
            key_required=bool(data.get("key_required")),
            key_name=_clean_resref(data.get("key_name")),
            auto_remove_key=bool(data.get("auto_remove_key")),
            unlock_dc=int(data.get("unlock_dc") or 0),
            lock_dc=int(data.get("lock_dc") or 0),
            trap_detectable=bool(data.get("trap_detectable")),
            trap_detect_dc=int(data.get("trap_detect_dc") or 0),
            trap_disarmable=bool(data.get("trap_disarmable")),
            trap_disarm_dc=int(data.get("trap_disarm_dc") or 0),
            trap_flag=int(data.get("trap_flag") or 0),
            trap_one_shot=bool(data.get("trap_one_shot")),
            trap_type=int(data.get("trap_type") or 0),
            maximum_hp=int(data.get("maximum_hp", 1)),
            current_hp=int(data.get("current_hp", data.get("maximum_hp", 1))),
            hardness=int(data.get("hardness") or 0),
            plot=bool(data.get("plot")),
            min1_hp=bool(data.get("min1_hp")),
            not_blastable=bool(data.get("not_blastable")),
            party_interact=bool(data.get("party_interact")),
            conversation_resref=_clean_resref(data.get("conversation_resref")),
        )


@dataclass
class PlaceableResourceRefs:
    mdl: ResourceAddress | None = None
    mdx: ResourceAddress | None = None
    pwk: ResourceAddress | None = None
    textures: list[ResourceAddress] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mdl": _address_dict(self.mdl),
            "mdx": _address_dict(self.mdx),
            "pwk": _address_dict(self.pwk),
            "textures": [value.to_dict() for value in self.textures],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PlaceableResourceRefs":
        data = data or {}
        return cls(
            mdl=_address(data.get("mdl")),
            mdx=_address(data.get("mdx")),
            pwk=_address(data.get("pwk")),
            textures=[ResourceAddress.from_dict(value) for value in data.get("textures") or ()],
        )


@dataclass
class PlaceableAsset:
    asset_id: str = field(default_factory=lambda: stable_project_id("placeable"))
    game: str = "K2"
    template_resref: str = ""
    tag: str = ""
    display_name: str = ""
    description: str = ""
    comment: str = ""
    category: str = "decor"
    visual_source: str = "stock"
    appearance_id: int | None = None
    gameplay: PlaceableGameplay = field(default_factory=PlaceableGameplay)
    scripts: dict[str, str] = field(default_factory=dict)
    resources: PlaceableResourceRefs = field(default_factory=PlaceableResourceRefs)
    base_template: ResourceAddress | None = None
    base_evidence: PlaceableBaseTemplateEvidence | None = None
    appearance_evidence: PlaceableAppearanceMappingEvidence | None = None
    secondary_motion: dict[str, Any] = field(default_factory=default_secondary_motion_profile)
    metadata: dict[str, Any] = field(default_factory=dict)
    extra_fields: dict[str, Any] = field(default_factory=dict, repr=False)
    schema_version: int = CURRENT_PLACEABLE_ASSET_SCHEMA_VERSION
    file_type: str = PLACEABLE_ASSET_FILE_TYPE

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.extra_fields)
        payload.update({
            "file_type": self.file_type,
            "schema_version": int(self.schema_version),
            "asset_id": self.asset_id,
            "game": str(self.game or "").upper(),
            "template_resref": _clean_resref(self.template_resref),
            "tag": str(self.tag or ""),
            "display_name": str(self.display_name or ""),
            "description": str(self.description or ""),
            "comment": str(self.comment or ""),
            "category": str(self.category or "").lower(),
            "visual_source": str(self.visual_source or "").lower(),
            "appearance_id": self.appearance_id,
            "gameplay": self.gameplay.to_dict(),
            "scripts": {key: _clean_resref(value) for key, value in sorted(self.scripts.items()) if value},
            "resources": self.resources.to_dict(),
            "base_template": _address_dict(self.base_template),
            "base_evidence": self.base_evidence.to_dict() if self.base_evidence else None,
            "appearance_evidence": self.appearance_evidence.to_dict() if self.appearance_evidence else None,
            "secondary_motion": SecondaryMotionProfile.from_dict(self.secondary_motion).to_dict(),
            "metadata": dict(self.metadata or {}),
        })
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlaceableAsset":
        if not isinstance(data, Mapping):
            raise ValueError("Placeable asset data must be a JSON object.")
        data = migrate_placeable_asset_payload(data)
        return cls(
            asset_id=str(data.get("asset_id") or stable_project_id("placeable")),
            game=str(data.get("game") or "K2").upper(),
            template_resref=_clean_resref(data.get("template_resref")),
            tag=str(data.get("tag") or ""),
            display_name=str(data.get("display_name") or ""),
            description=str(data.get("description") or ""),
            comment=str(data.get("comment") or ""),
            category=str(data.get("category") or "decor").lower(),
            visual_source=str(data.get("visual_source") or "stock").lower(),
            appearance_id=int(data["appearance_id"]) if data.get("appearance_id") is not None else None,
            gameplay=PlaceableGameplay.from_dict(data.get("gameplay")),
            scripts={str(key): _clean_resref(value) for key, value in dict(data.get("scripts") or {}).items()},
            resources=PlaceableResourceRefs.from_dict(data.get("resources")),
            base_template=_address(data.get("base_template")),
            base_evidence=PlaceableBaseTemplateEvidence.from_dict(data.get("base_evidence")),
            appearance_evidence=PlaceableAppearanceMappingEvidence.from_dict(data.get("appearance_evidence")),
            secondary_motion=SecondaryMotionProfile.from_dict(data.get("secondary_motion")).to_dict(),
            metadata=dict(data.get("metadata") or {}),
            extra_fields={
                str(key): value
                for key, value in data.items()
                if key not in {
                    "file_type", "schema_version", "asset_id", "game", "template_resref", "tag",
                    "display_name", "description", "comment", "category", "visual_source",
                    "appearance_id", "gameplay", "scripts", "resources", "base_template",
                    "base_evidence", "appearance_evidence", "secondary_motion", "metadata",
                }
            },
            schema_version=int(data.get("schema_version") or 0),
            file_type=str(data.get("file_type") or ""),
        )


def migrate_placeable_asset_payload(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Migrate placeable authoring documents without discarding extensions."""

    if not isinstance(raw, Mapping):
        raise ValueError("Placeable asset data must be a JSON object.")
    data = dict(raw)
    version = int(data.get("schema_version") or 0)
    if version > CURRENT_PLACEABLE_ASSET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported placeable asset schema version {version}.")
    if version in {0, 1}:
        data.setdefault("file_type", PLACEABLE_ASSET_FILE_TYPE)
        data.setdefault(
            "secondary_motion",
            default_secondary_motion_profile(target_game=str(data.get("game") or "K2")),
        )
        data["schema_version"] = 2
    if data.get("file_type") != PLACEABLE_ASSET_FILE_TYPE:
        raise ValueError("File is not a GhostRigger placeable asset.")
    if int(data.get("schema_version") or 0) != CURRENT_PLACEABLE_ASSET_SCHEMA_VERSION:
        raise ValueError("Placeable asset schema is not supported.")
    return data


@dataclass(frozen=True)
class PlaceableAssetValidation:
    issues: tuple[ProjectValidationIssue, ...]
    document_valid: bool
    utp_export_ready: bool
    structural_evidence_ready: bool
    engine_ready: bool = False


def _valid_resref(value: str) -> bool:
    return bool(value and len(value) <= 16 and _RESREF_RE.fullmatch(value))


def validate_placeable_asset(asset: PlaceableAsset) -> PlaceableAssetValidation:
    report = ProjectValidationReport()
    if asset.file_type != PLACEABLE_ASSET_FILE_TYPE:
        report.add("blocking", "wrong_placeable_file_type", "Placeable asset file_type is not recognized.", target="file_type")
    if asset.schema_version != CURRENT_PLACEABLE_ASSET_SCHEMA_VERSION:
        report.add("blocking", "unsupported_placeable_schema", "Placeable asset schema version is unsupported.", target="schema_version")
    if not _ASSET_ID_RE.fullmatch(asset.asset_id):
        report.add("blocking", "invalid_placeable_asset_id", "Placeable asset ID must remain a stable placeable_<uuid> identity.", target="asset_id")
    if asset.game not in {"K1", "K2"}:
        report.add("blocking", "invalid_placeable_game", "Placeable asset game must be K1 or K2.", target="game")
    if not _valid_resref(_clean_resref(asset.template_resref)):
        report.add("blocking", "invalid_placeable_resref", "Template resref must use 1-16 letters, numbers, or underscores.", target="template_resref")
    if not asset.tag.strip():
        report.add("blocking", "missing_placeable_tag", "Placeable tag is required.", target="tag")
    if asset.category not in PLACEABLE_CATEGORIES:
        report.add("blocking", "invalid_placeable_category", f"Category must be one of {', '.join(PLACEABLE_CATEGORIES)}.", target="category")
    if asset.visual_source not in {"stock", "custom"}:
        report.add("blocking", "invalid_visual_source", "Visual source must be stock or custom.", target="visual_source")
    if asset.appearance_id is None or asset.appearance_id < 0:
        report.add("blocking", "missing_placeable_appearance", "UTP export requires a non-negative placeables.2da Appearance row.", target="appearance_id")

    gameplay = asset.gameplay
    if gameplay.locked and not gameplay.lockable:
        report.add("blocking", "locked_not_lockable", "A locked placeable must also be lockable.", target="gameplay.locked")
    if gameplay.key_required and (not gameplay.lockable or not _valid_resref(gameplay.key_name)):
        report.add("blocking", "invalid_placeable_key", "Key-required placeables need a lockable flag and valid key resref.", target="gameplay.key_name")
    if gameplay.has_inventory is False and gameplay.inventory_items:
        report.add("blocking", "inventory_flag_missing", "Inventory items require has_inventory.", target="gameplay.inventory_items")
    for index, item in enumerate(gameplay.inventory_items):
        if not _valid_resref(item):
            report.add("blocking", "invalid_inventory_resref", "Inventory item resrefs must be KOTOR-safe.", target=f"gameplay.inventory_items.{index}")
    for name, value in (("maximum_hp", gameplay.maximum_hp), ("current_hp", gameplay.current_hp)):
        if not -32768 <= int(value) <= 32767:
            report.add("blocking", "placeable_hp_out_of_range", f"{name} must fit the UTP INT16 field.", target=f"gameplay.{name}")
    for name in ("unlock_dc", "lock_dc", "hardness", "trap_detect_dc", "trap_disarm_dc", "trap_flag", "trap_type"):
        if not 0 <= int(getattr(gameplay, name)) <= 255:
            report.add("blocking", "placeable_byte_out_of_range", f"{name} must fit the UTP BYTE field.", target=f"gameplay.{name}")
    if gameplay.conversation_resref and not _valid_resref(gameplay.conversation_resref):
        report.add("blocking", "invalid_conversation_resref", "Conversation resref must be KOTOR-safe.", target="gameplay.conversation_resref")
    for hook, script in asset.scripts.items():
        if hook not in PLACEABLE_SCRIPT_HOOKS:
            report.add("blocking", "unknown_placeable_script_hook", f"Unsupported placeable script hook '{hook}'.", target=f"scripts.{hook}")
        elif script and not _valid_resref(script):
            report.add("blocking", "invalid_script_resref", "Script resrefs must be KOTOR-safe.", target=f"scripts.{hook}")

    typed_resources = (("mdl", asset.resources.mdl, "MDL"), ("mdx", asset.resources.mdx, "MDX"), ("pwk", asset.resources.pwk, "PWK"))
    for name, address, expected in typed_resources:
        if address is None:
            continue
        report.issues.extend(validate_resource_address(address))
        if address.restype != expected:
            report.add("blocking", "wrong_placeable_resource_type", f"{name} reference must use {expected}.", target=address)
    for address in asset.resources.textures:
        report.issues.extend(validate_resource_address(address))
        if address.restype not in {"TPC", "TGA", "TXI"}:
            report.add("blocking", "wrong_placeable_texture_type", "Texture references must be TPC, TGA, or TXI.", target=address)
    if asset.visual_source == "custom" and (asset.resources.mdl is None or asset.resources.mdx is None):
        report.add("blocking", "custom_placeable_model_pair_missing", "Custom visuals require paired MDL and MDX resources.", target="resources")

    if asset.base_template is not None:
        report.issues.extend(validate_resource_address(asset.base_template))
        if asset.base_template.restype != "UTP":
            report.add("blocking", "wrong_base_template_type", "Base template reference must use UTP.", target=asset.base_template)

    base_ready = bool(
        asset.base_template
        and asset.base_evidence
        and asset.base_evidence.template
        and asset.base_evidence.template.stable_key() == asset.base_template.stable_key()
        and re.fullmatch(r"[0-9a-f]{64}", asset.base_evidence.sha256 or "")
        and asset.base_evidence.field_count > 0
    )
    appearance_ready = bool(
        asset.appearance_evidence
        and asset.appearance_evidence.verified
        and asset.appearance_evidence.game == asset.game
        and asset.appearance_evidence.appearance_id == asset.appearance_id
        and _valid_resref(asset.appearance_evidence.model_resref)
        and asset.appearance_evidence.source_sha256
    )
    if not base_ready:
        report.add("warning", "placeable_base_evidence_missing", "No byte-verified vanilla/base UTP structural authority is attached.", target="base_evidence")
    if not appearance_ready:
        report.add("warning", "placeable_appearance_mapping_unverified", "Appearance row to model mapping is not byte/source verified.", target="appearance_evidence")

    document_valid = not report.has_blocking
    return PlaceableAssetValidation(
        issues=tuple(report.issues),
        document_valid=document_valid,
        utp_export_ready=document_valid and asset.appearance_id is not None,
        structural_evidence_ready=document_valid and base_ready and appearance_ready,
        engine_ready=False,
    )


def save_placeable_asset(asset: PlaceableAsset, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(asset.to_dict(), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
    return target


def load_placeable_asset(path: str | Path) -> PlaceableAsset:
    return PlaceableAsset.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = [
    "CURRENT_PLACEABLE_ASSET_SCHEMA_VERSION",
    "PLACEABLE_ASSET_FILE_SUFFIX",
    "PLACEABLE_ASSET_FILE_TYPE",
    "PLACEABLE_CATEGORIES",
    "PLACEABLE_SCRIPT_HOOKS",
    "PlaceableAppearanceMappingEvidence",
    "PlaceableAsset",
    "PlaceableAssetValidation",
    "PlaceableBaseTemplateEvidence",
    "PlaceableGameplay",
    "PlaceableResourceRefs",
    "load_placeable_asset",
    "migrate_placeable_asset_payload",
    "save_placeable_asset",
    "validate_placeable_asset",
]
