"""Project-local immutable assets for Unreal imports. Core.IO owns all disk IO."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.io.unreal_level_bundle import MAX_JSON_BYTES, _json_object, load_unreal_level_bundle, read_bundle_file


class UnrealProjectAssets:
    def __init__(self, project_directory: str | Path):
        self.root = Path(project_directory).resolve()

    def put(self, data: bytes, suffix: str) -> dict:
        if len(data) > MAX_JSON_BYTES:
            raise ValueError("Project import asset exceeds the bounded sidecar size.")
        if suffix not in {"json", "tga", "umap"}:
            raise ValueError("Unsupported immutable import asset type.")
        digest = hashlib.sha256(data).hexdigest()
        relative = Path("unreal_assets") / f"{digest}.{suffix}"
        final = self.root / relative
        if not final.resolve().is_relative_to(self.root):
            raise ValueError("Project asset directory escapes the project root.")
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if hashlib.sha256(final.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Project asset was altered: {relative}")
        else:
            descriptor, name = tempfile.mkstemp(prefix=".import-", dir=final.parent)
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(temporary, final)  # Atomic no-overwrite publication.
                except FileExistsError:
                    if hashlib.sha256(final.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Concurrent import asset conflict: {relative}")
            finally:
                temporary.unlink(missing_ok=True)
        return {"path": relative.as_posix(), "sha256": digest}

    def put_json(self, value: dict) -> dict:
        return self.put((json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(), "json")

    def read(self, reference: dict) -> bytes:
        return read_bundle_file(self.root, reference)

    def read_json(self, reference: dict) -> dict:
        return _json_object(self.read(reference))

    def capture_bundle(self, path: str | Path, *, expected_revision: str | None = None) -> tuple[dict, dict, str]:
        source = Path(path).resolve()
        if source.stat().st_size > MAX_JSON_BYTES:
            raise ValueError("Import manifest exceeds the bounded sidecar size.")
        if self.root == source.parent or source.parent.is_relative_to(self.root) or self.root.is_relative_to(source.parent):
            raise ValueError("Project asset storage must be separate from the source bundle.")
        original_bytes = source.read_bytes()
        source_revision = hashlib.sha256(original_bytes).hexdigest()
        if expected_revision is not None and expected_revision != source_revision:
            raise ValueError("Source changed after preview; inspect again.")
        # Fully validate/decode first. No published scene state on failure.
        load_unreal_level_bundle(source)
        manifest = _json_object(original_bytes)

        def copy_reference(reference, suffix):
            return self.put(read_bundle_file(source.parent, reference), suffix)

        manifest["source"]["map_file"] = copy_reference(manifest["source"]["map_file"], "umap")
        for actor in manifest["actors"]:
            if "mesh_file" in actor:
                actor["mesh_file"] = copy_reference(actor["mesh_file"], "json")
        for material in manifest["materials"]:
            material["diffuse_file"] = copy_reference(material["diffuse_file"], "tga")
        if source.read_bytes() != original_bytes:
            raise ValueError("Source manifest changed while importing.")
        return self.put_json(manifest), manifest, source_revision

    def resolve_bundle(self, manifest: dict) -> dict:
        """Resolve rebased refs against the project root (not the manifest folder)."""
        result = copy.deepcopy(manifest)
        self.read(result["source"]["map_file"])
        for actor in result["actors"]:
            if "mesh_file" in actor:
                actor["mesh"] = self.read_json(actor["mesh_file"])
        for material in result["materials"]:
            material["diffuse_bytes"] = self.read(material["diffuse_file"])
        return result
