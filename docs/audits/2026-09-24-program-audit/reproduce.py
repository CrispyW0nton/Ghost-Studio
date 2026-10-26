"""Bounded audit probes. All destructive/failure probes use temporary files only.

Run from the repository: python docs/audits/2026-09-24-program-audit/reproduce.py
These record observed defects, not passing acceptance tests or native UI proof.
"""
from __future__ import annotations

import ast
import importlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys
import tempfile
import tomllib
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def export_probes():
    import src.core.export.export_job as ex

    observed = {}
    with tempfile.TemporaryDirectory(prefix="ghost-audit-") as temporary:
        root = Path(temporary)
        paths = [root / "a/same.bin", root / "b/same.bin", root / "c/trigger.bin"]
        for index, path in enumerate(paths):
            path.parent.mkdir()
            path.write_text(f"original-{index}")
        original_replace = ex.os.replace

        def fail_third(source, destination):
            if Path(destination) == paths[2] and not str(source).endswith("ghostrigger_backup"):
                raise OSError("injected promotion failure")
            return original_replace(source, destination)

        def writer(context):
            for path in paths:
                context.write_text(path, "new")

        request = ex.ExportJobRequest("audit", "files", [ex.ExportOutputSpec(p, "data") for p in paths], overwrite=True)
        with patch.object(ex.os, "replace", fail_third):
            result = ex.run_export_job(request, writer=writer)
        observed["basename_backup_collision"] = {
            "status": result.status.value,
            "expected": [f"original-{i}" for i in range(3)],
            "actual": [p.read_text() if p.exists() else "MISSING" for p in paths],
        }
        target = root / "locked.bin"
        target.write_text("irreplaceable original")

        def destination_locked(source, destination):
            if Path(destination) == target:
                raise PermissionError("injected destination lock")
            return original_replace(source, destination)

        request = ex.ExportJobRequest("restore_audit", "files", [ex.ExportOutputSpec(target, "data")], overwrite=True)
        with patch.object(ex.os, "replace", destination_locked):
            result = ex.run_export_job(request, writer=lambda c: c.write_text(target, "new"))
        observed["failed_restore_discards_backup"] = {
            "status": result.status.value,
            "original_exists": target.exists(),
            "recoverable_backups": len(list(root.rglob("*.ghostrigger_backup"))),
            "reported_rollback_error": any("Rollback issues" in i.message for i in result.validation_report.issues),
        }
        required, optional = root / "required.bin", root / "optional.bin"
        request = ex.ExportJobRequest("optional_audit", "files", [ex.ExportOutputSpec(required, "data"), ex.ExportOutputSpec(optional, "data", required=False)])
        result = ex.run_export_job(request, writer=lambda c: c.write_text(required, "new"))
        observed["omitted_optional_output"] = {"status": result.status.value, "required_published": required.exists()}
    return observed


def scene_save_probe():
    from src.core.scene.kmax_scene import KMaxScene
    from src.core.scene.kmax_serializer import KMaxSerializer

    with tempfile.TemporaryDirectory(prefix="ghost-audit-save-") as temporary:
        target = Path(temporary) / "scene.kmax"
        scene = KMaxScene.new()
        KMaxSerializer.save(scene, target)
        old = target.read_bytes()
        scene.name = "edited"
        scene.mark_dirty()
        original_write = Path.write_text

        def partial_write(path, data, *args, **kwargs):
            if path == target:
                with path.open("w", encoding="utf-8") as stream:
                    stream.write(data[:20])
                raise OSError("injected disk full")
            return original_write(path, data, *args, **kwargs)

        try:
            with patch.object(Path, "write_text", partial_write):
                KMaxSerializer.save(scene, target)
        except OSError:
            pass
        return {"previous_save_bytes": len(old), "remaining_bytes": len(target.read_bytes()), "previous_save_preserved": old == target.read_bytes()}


def payload_order_probe():
    # Execute the actual importer class only; do not launch the host or load DLLs.
    host = ROOT / "native/GhostRigger.Native.Core.Host/main.py"
    node = next(n for n in ast.parse(host.read_text(encoding="utf-8")).body if isinstance(n, ast.ClassDef) and n.name == "_DllPythonPayloadImporter")
    namespace = {"importlib": importlib, "Path": Path}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(host), "exec"), namespace)
    importer_type = namespace[node.name]

    class ManifestFunction:
        def __init__(self, project):
            self.project = project

        def __call__(self):
            return (ROOT / "native" / self.project / "GhostRiggerPythonPayload.json").read_bytes()

    observations = []
    for order in [("GhostRigger.Core.IO", "GhostRigger.Core.Tools"), ("GhostRigger.Core.Tools", "GhostRigger.Core.IO")]:
        importer = importer_type.__new__(importer_type)
        importer._build_dir = SimpleNamespace(glob=lambda _: [Path(p + ".dll") for p in order])
        importer._asset_root = Path("unused")
        importer._modules, importer._packages, importer._namespace_packages, importer._dlls = {}, set(), {}, []
        importer._ctypes = SimpleNamespace(c_char_p=None, CDLL=lambda path, winmode: SimpleNamespace(project=Path(path).stem, gr_python_payload_manifest_json=ManifestFunction(Path(path).stem)))
        importer._load_manifests()
        owner = importer._modules["src.core.export.export_job"]["dll"].project
        source = ROOT / "native" / owner / "Python/src/core/export/export_job.py"
        observations.append({"order": order, "provider": owner, "has_promote_staged_files": "def promote_staged_files(" in source.read_text()})
    return observations


def model_pair_probe():
    from src.core.assets.resource_manager import RES_MDL, RES_MDX, ResourceManager
    from src.core.game import kotor_loader

    manager = ResourceManager()
    manager.set_project_overlay((("bench", RES_MDL, b"old-mdl"), ("bench", RES_MDX, b"old-mdx")))
    original_get = manager.get_mdl

    def publish_between_reads(name, game):
        data = original_get(name, game)
        manager.set_project_overlay((("bench", RES_MDL, b"new-mdl"), ("bench", RES_MDX, b"new-mdx")))
        return data

    manager.get_mdl = publish_between_reads
    pairs = []
    with patch.object(kotor_loader, "load_model_from_bytes", lambda mdl, mdx: (pairs.append([mdl.decode(), mdx.decode()]) or SimpleNamespace())):
        manager.load_model("bench", "K1")
    return pairs


def main():
    result = export_probes()
    result["non_atomic_kmax_save"] = scene_save_probe()
    result["dll_discovery_order"] = payload_order_probe()
    result["mixed_model_revision"] = model_pair_probe()
    backend = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["build-backend"]
    try:
        importlib.import_module(backend.split(":")[0])
        result["build_backend"] = {"backend": backend, "importable": True}
    except ImportError as error:
        result["build_backend"] = {"backend": backend, "importable": False, "error": str(error)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
