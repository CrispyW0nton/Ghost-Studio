"""Preview/build a versioned Unreal interchange bundle using Ghost Studio.

This CLI does not run Unreal, install game files or claim retail/UI acceptance.
Preview a loose .umap to see dependency hints; conversion needs a native bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io.unreal_level_bundle import inspect_loose_umap, load_unreal_level_bundle
from src.core.modules.unreal_level_import import build_unreal_level_project


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preview", "build"))
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-sha256", help="Manifest revision returned by preview; required for build.")
    args = parser.parse_args(argv)
    try:
        if args.source.suffix.lower() == ".umap":
            result = inspect_loose_umap(args.source)
            if args.mode == "build":
                raise ValueError(result["next_step"])
            print(json.dumps(result, indent=2))
            return 0
        bundle = load_unreal_level_bundle(args.source)
        imported = build_unreal_level_project(bundle)
        from src.core.modules.authored_module_project import validate_authored_module_project
        validation = validate_authored_module_project(imported.project)
        summary = {"schema": "ghoststudio.unreal-level-build/v1", "manifest_sha256": bundle["bundle_sha256"],
                   "module": imported.project.module_root, "rooms": len(imported.project.rooms),
                   "warnings": list(imported.warnings) + list(validation.warnings),
                   "blocking_issues": list(validation.blocking_issues), "game_tested": False}
        if args.mode == "preview":
            summary["ok"] = validation.ok
            print(json.dumps(summary, indent=2))
            return 0 if validation.ok else 2
        if not validation.ok:
            raise ValueError("Source geometry failed validation: " + "; ".join(validation.blocking_issues))
        if args.expected_sha256 != bundle["bundle_sha256"]:
            raise ValueError("Expected manifest revision does not match; preview again.")
        if args.output is None:
            raise ValueError("Build requires a new output directory.")
        output = args.output.resolve()
        source_root = args.source.resolve().parent
        if output == source_root or output.is_relative_to(source_root) or source_root.is_relative_to(output):
            raise ValueError("Build output must be separate from the source bundle.")
        output.mkdir(parents=True, exist_ok=False)
        from src.core.game.pykotor_gff_runtime_fix import ensure_pykotor_gff_acquire_quiet
        from src.core.modules.authored_module_export import AuthoredModuleExportRequest, export_authored_module_project
        ensure_pykotor_gff_acquire_quiet()
        result = export_authored_module_project(AuthoredModuleExportRequest(
            project=imported.project, output_dir=str(output / "package"),
            extra_resources=imported.extra_resources, strict=True))
        summary.update(ok=result.ok, module_path=result.module_path,
                       blocking_issues=list(result.blocking_issues), warnings=summary["warnings"]+list(result.warnings))
        if result.ok:
            module = Path(result.module_path)
            summary["module_sha256"] = hashlib.sha256(module.read_bytes()).hexdigest()
            from src.core.level import new_kmap_project, TextureReference
            from src.core.level.kmap_serializer import KMapSerializer
            from src.core.modules.authored_module_kmap_bridge import authored_project_to_kmap_payload
            project = new_kmap_project(name=imported.project.module_root, game="K2")
            project.extra_sections["authored_module"] = authored_project_to_kmap_payload(imported.project, game_tested=False)
            # Keep only references in the texture table; no new blob format.
            textures = output / "textures"; textures.mkdir()
            for name, kind, data in imported.extra_resources:
                texture_path = textures / f"{name}.{kind}"
                with texture_path.open("xb") as stream: stream.write(data)
                if kind == "tga":
                    project.textures.append(TextureReference(resref=name,path=str(texture_path),source="unreal_level_bundle"))
            kmap = output / f"{imported.project.module_root}.kmap"
            KMapSerializer.save(project, kmap)
            summary["kmap"] = str(kmap)
        with (output / "receipt.json").open("x",encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        print(json.dumps(summary, indent=2))
        return 0 if result.ok else 2
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"ok": False, "error": str(error), "game_tested": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
