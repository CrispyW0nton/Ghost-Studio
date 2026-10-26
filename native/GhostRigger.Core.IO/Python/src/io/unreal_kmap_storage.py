"""Compact Unreal-backed KMAP documents and failure-safe JSON publication."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import tempfile
from .unreal_project_assets import UnrealProjectAssets


def hydrate_unreal_document(data: dict, source: Path) -> dict:
    authored = data.get("authored_module", {})
    if not isinstance(authored, dict):
        raise ValueError("Malformed authored project reference.")
    if "schema" in authored and authored["schema"] != "ghoststudio.authored-reference/v1":
        raise ValueError("Unsupported authored project reference version.")
    if authored.get("schema") != "ghoststudio.authored-reference/v1":
        return data
    if not data.get("unreal_source"):
        raise ValueError("An authored source reference requires its Unreal project contract.")
    result = copy.deepcopy(data)
    result["authored_module"] = UnrealProjectAssets(source.resolve().parent).read_json(authored["file"])
    return result


def persist_unreal_document(data: dict, target: Path, original_path: str) -> None:
    target = target.resolve()
    if original_path and Path(original_path).resolve().parent != target.parent:
        raise ValueError("Save this Unreal KMAP beside its assets; move the complete project folder to relocate it.")
    document = copy.deepcopy(data)
    assets = UnrealProjectAssets(target.parent)
    document["authored_module"] = {"schema": "ghoststudio.authored-reference/v1", "file": assets.put_json(document["authored_module"])}
    encoded = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=".kmap-save-", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
